import json
import glob
import time
from tqdm import tqdm
import os
from MMLLM_Common import *
from google.api_core.exceptions import ResourceExhausted, InternalServerError, TooManyRequests, ServerError, BadRequest
import google.generativeai as genai


class MMLLM_Gemini:
    def __init__(self, str_api_key:str):
        self.str_api_key = str_api_key
        self.str_phase1_system_msg:dict        
        self.str_phase1_res_format:dict
        self.str_phase2_system_msg:dict        
        genai.configure(api_key=str_api_key)

        self.str_model = "gemini-1.5-flash"
        self.client = genai.GenerativeModel(self.str_model)
        return    

    def load_prompt_text(self, input_mode:InputMode):        
        # Prompts
        str_phase1_prompt_path = dict_system_prompt_path.get(input_mode)
        assert str_phase1_prompt_path is not None, f"Unknown Input mode {input_mode}"

        with open(str_phase1_prompt_path, encoding='utf-8') as f_read:
            str_phase1_system_prompt = f_read.read()        
            self.str_phase1_system_msg = str_phase1_system_prompt

        # Response        
        str_phase1_response_prompt_path = dict_response_prompt_path.get(input_mode)
        assert str_phase1_response_prompt_path is not None, f"Unknown Input mode {input_mode}"

        with open(str_phase1_response_prompt_path, encoding='utf-8') as f_read:
            str_res_format = f_read.read()
            self.str_phase1_res_format = '\n\n' + str_res_format

        # Phase 2 Prompt
        str_phase2_system_prompt_path = dict_response_prompt_path.get(Phase2Mode.Phase2)
        assert str_phase2_system_prompt_path is not None, f"Unknown Input mode {Phase2Mode.Phase2}"

        with open(str_phase2_system_prompt_path, encoding='utf-8') as f_read:
            str_phase2_system_prompt = f_read.read()
            self.str_phase2_system_msg = str_phase2_system_prompt
        return

    def create_identification_prompt(self, input_mode:InputMode, encoded_image, html_content):                
        '''
        First phase: Prompt for Brand identification. Gemini format.
        '''      
        str_resource_msg = "Here are the provided resources: "
        list_phase1_system_msg = [self.str_phase1_system_msg, str_resource_msg]

        if input_mode == InputMode.SS:
           # Add Image
           list_phase1_system_msg.append(encoded_image)
        elif input_mode == InputMode.HTML:
           list_phase1_system_msg.append(html_content)
        elif input_mode == InputMode.BOTH:
            list_phase1_system_msg.append(html_content)
            list_phase1_system_msg.append(encoded_image)
        
        list_phase1_system_msg.append(self.str_phase1_res_format)

        return list_phase1_system_msg

    def create_brandcheck_prompt(self, str_groundtruth:str, str_prediction:str):                
        '''
        Second phase: Prompt for Brand comparison and Phishing classification. Gemini format.
        '''

        str_phase2_data = f"Ground Truth: \"{str_groundtruth}\"\n\"Prediction:\"{str_prediction}\""
        
        list_phase2_system_msg = [self.str_phase2_system_msg, str_phase2_data]
        
        return list_phase2_system_msg
    

    def phase1_brand_identification(self, input_dataset:InputDataset):
        str_dataset = input_dataset
        list_data_dir = glob.glob(f'{str_input_dir_base}/{str_dataset}/*/*/') # ../input/<dataset>/<brand>/<hash>/
        list_data_dir.sort()

        for str_data_dir in tqdm(list_data_dir, desc=f'{str_dataset}'):
            str_data_dir = str_data_dir.replace('\\', '/')
            list_prop = str_data_dir.split('/')
            str_ss_path = str_data_dir + '/screenshot_aft.png'
            str_html_path = str_data_dir + '/add_info.json'

            str_hash = list_prop[-2]
            str_brand = list_prop[-3]

            if not os.path.exists(str_ss_path) or not os.path.exists(str_html_path):
                # Error sample
                continue
        
            image = crop_encode_image_PIL(str_ss_path)
            
            # HTML Input
            with open(str_html_path) as f_read:            
                str_html_info = json.load(f_read)['html_brand_info']            

            for input_mode in InputMode:                
                self.load_prompt_text(input_mode) # Following input_mode

                str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Gemini', input_mode, str_brand)
                if not os.path.exists(str_output_dir):
                    os.makedirs(str_output_dir)    
                
                # Gemini
                list_model_prompt = self.create_identification_prompt(input_mode, image, str_html_info)
                
                try:
                    response = self.client.generate_content(list_model_prompt)
                    response.resolve()
                except ResourceExhausted:
                    # Error if Quota limited.
                    print(f'[Warning] {str_hash} made Quota error')
                    time.sleep(60*60)            
                    continue
                except InternalServerError:                    
                    print(f'[Warning] {str_hash} InternalServerError')
                    time.sleep(60*60)            
                    continue
                except BadRequest:
                    print(f'[Warning] {str_hash} BadRequest')
                    continue
                        
                if response.prompt_feedback.block_reason != 0:
                    dict_res_data = format_model_response(str_hash, '', True, False)                                                                 
                elif not response.parts:
                    str_res = response.candidates
                    dict_res_data = format_model_response(str_hash, str_res, False, True)
                elif response.parts == []:
                    dict_res_data = format_model_response(str_hash, '', False, True)
                else:                    
                    # Normal
                    str_res : str
                    str_res = response.text                    

                    try:
                        dict_res_data = format_model_response(str_hash, str_res)                
                    except:
                        dict_res_data = format_model_response(str_hash, 'Safety Error', False, True)                        
                        pass

                if response.usage_metadata:
                    dict_res_data['prompt_token_count'] = response.usage_metadata.prompt_token_count
                    dict_res_data['candidates_token_count'] = response.usage_metadata.candidates_token_count
                    dict_res_data['total_token_count'] = response.usage_metadata.total_token_count
                    
                                   
                
                str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                with open(str_output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(dict_res_data, f, indent=4)
        return
    

    def phase2_phishing_classification(self, input_dataset:InputDataset):
        str_dataset = input_dataset
        # No need to Load Phase2 Prompt. It is already loaded in the instance at Phase1.
        
        # Output: Summary
        str_output_summary_path = os.path.join(str_output_dir_base, str_dataset, 'Phase2_Gemini', "Phase2_Res_Summary.csv")
        if not os.path.exists(str_output_summary_path):
            with open(str_output_summary_path, 'w') as f_summary:
                str_phase2_res_summary_hdr = f'Dataset,InputMode,Brand,Hash,Phase1Pred,Phase2Matched\n'
                f_summary.write(str_phase2_res_summary_hdr)

        for input_mode in InputMode:
            # Input: Came from Output of Phase 1
            str_input_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Gemini', input_mode, str_brand)                 
            list_input_path = glob.glob(f'{str_input_dir}/*.json') # ./output/<dataset>/<input_mode>/<brand>/<hash>.json
            list_input_path = [x.replace('\\', '/') for x in list_input_path]
            list_input_path.sort()

            
            str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase2_Gemini', input_mode, str_brand)
            if not os.path.exists(str_output_dir):
                os.makedirs(str_output_dir)

            for str_input_path in list_input_path:
                list_prop = str_input_path.split('/')

                str_hash = list_prop[-1].replace('.json', '')
                str_brand = list_prop[-2]
                                              
                # Brand Prediction at Phase 1
                try:
                    with open(str_input_path) as f_read:
                        str_phase1_pred = json.load(f_read)['Brand']            
                        b_phase1_error = json.load(f_read)['Error']                        
                except:
                    print(f'[Warning] Broken Phase 1 result: {str_input_path}')
                    continue
            
                if b_phase1_error == True:
                    continue

                # Gemini
                list_model_prompt = self.create_brandcheck_prompt(input_mode, str_brand, str_phase1_pred)
               
                try:
                    response = self.client.generate_content(list_model_prompt)
                    response.resolve()
                except ResourceExhausted:
                    # Error if Quota limited.
                    print(f'[Warning] {str_hash} made Quota error')
                    time.sleep(60*60)            
                    continue
                except InternalServerError:                    
                    print(f'[Warning] {str_hash} InternalServerError')
                    time.sleep(60*60)            
                    continue
                except BadRequest:
                    print(f'[Warning] {str_hash} BadRequest')
                    continue
                        
                if response.prompt_feedback.block_reason != 0:
                    dict_res_data = format_phase2_response('', True, False)                                                                 
                elif not response.parts:
                    str_res = response.candidates
                    dict_res_data = format_phase2_response(str_res, False, True)
                elif response.parts == []:
                    dict_res_data = format_phase2_response('', False, True)
                else:                    
                    # Normal
                    str_res : str
                    str_res = response.text                    

                    try:
                        dict_res_data = format_phase2_response(str_res)                
                    except:
                        dict_res_data = format_phase2_response(str_res, False, True)                        
                        pass
                
                if response.usage_metadata:
                    dict_res_data['prompt_token_count'] = response.usage_metadata.prompt_token_count
                    dict_res_data['candidates_token_count'] = response.usage_metadata.candidates_token_count
                    dict_res_data['total_token_count'] = response.usage_metadata.total_token_count
                                    
                str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")                    
                with open(str_output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(dict_res_data, f, indent=4)

                str_phase2_res_summary = f'{str_dataset},{input_mode},{str_brand},{str_phase1_pred},{dict_res_data['BrandMatched']}\n'
                with open(str_output_summary_path, 'a') as f_summary:
                    f_summary.write(str_phase2_res_summary)
        return
    

