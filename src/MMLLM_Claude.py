import json
import glob
import time
from tqdm import tqdm
import os
from MMLLM_Common import *
import anthropic
from anthropic import Anthropic
import random

class MMLLM_Claude:
    def __init__(self, str_api_key:str):
        self.str_api_key = str_api_key
        self.str_phase1_system_msg:dict        
        self.str_phase1_res_format:dict
        self.str_phase2_system_msg:dict                

        self.str_model = "claude-3-opus-20240229"
        self.client = client = Anthropic(api_key=str_api_key)
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

    def create_identification_prompt(self, input_mode:InputMode, str_html_info, base64_image, str_image_media_type:str) -> list[dict]:
                                     
        '''
        First phase: Prompt for Brand identification. Gemini format.
        '''
        int_max_text = 4096
        dict_phase1_system_msg = {}
        dict_phase1_system_msg['role'] = 'user' 
        dict_phase1_system_msg['content'] = []
        
        dict_prompt = {}
        dict_prompt['type'] = 'text'    
        dict_prompt['text'] = f"{self.str_phase1_system_msg}\n\n{self.str_phase1_res_format}"
        
        if input_mode == InputMode.SS or input_mode == InputMode.BOTH:
            dict_image = {}
            dict_image['type'] = 'image'
            dict_image['source'] = {}
            dict_image['source']['type'] = 'base64'
            dict_image['source']['media_type'] = str_image_media_type
            dict_image['source']['data'] = base64_image
            dict_phase1_system_msg['content'].append(dict_image)            
        
        if input_mode == InputMode.HTML or input_mode == InputMode.BOTH:        
            dict_info = {}
            dict_info['type'] = 'text'    
            dict_info['text'] = f'This is the HTML info: {str_html_info[:int_max_text]}'    
            dict_phase1_system_msg['content'].append(dict_info)

        dict_phase1_system_msg['content'].append(dict_prompt)
        return [dict_phase1_system_msg]
      

    def create_brandcheck_prompt(self, str_groundtruth:str, str_prediction:str) -> list:                
        '''
        Second phase: Prompt for Brand comparison and Phishing classification. Gemini format.
        '''
        
        dict_phase2_system_msg = {}
        dict_phase2_system_msg['role'] = 'user' 
        dict_phase2_system_msg['content'] = []
        
        str_phase2_data = f"Ground Truth: \"{str_groundtruth}\"\n\"Prediction:\"{str_prediction}\""
        
        dict_prompt = {}
        dict_prompt['type'] = 'text'    
        dict_prompt['text'] = f"{self.str_phase2_system_msg}"
        
        dict_info = {}
        dict_info['type'] = 'text'    
        dict_info['text'] = f'{str_phase2_data}'    
        dict_phase2_system_msg['content'].append(dict_info)        
      
        return [dict_phase2_system_msg]
    

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

            if 'webp' in str_ss_path:
                image_media_type = "image/webp"
            elif 'jpg' in str_ss_path:
                image_media_type = "image/jpg"
            else:
                image_media_type = "image/png"
            base64_image = crop_encode_image_base64(str_ss_path)
            
            # HTML Input
            with open(str_html_path) as f_read:            
                str_html_info = json.load(f_read)['html_brand_info']            

            for input_mode in InputMode:                
                self.load_prompt_text(input_mode) # Following input_mode
                
                # Claude
                list_model_prompt = self.create_identification_prompt(input_mode, str_html_info, base64_image, image_media_type)
                
                try:
                    response = self.client.messages.create(
                        model=self.str_model,
                        max_tokens=4096,
                        messages=list_model_prompt
                    )

                    if not response.content:
                        # Error                
                        dict_res_data = format_model_response(str_hash, 'Error', False, True)
                    else:
                        # Normal
                        str_res = response.content[0].text
                        dict_res_data = format_model_response(str_hash, str_res)

                    if response.usage:            
                        dict_res_data['prompt_token_count'] = response.usage.input_tokens
                        dict_res_data['candidates_token_count'] = response.usage.output_tokens
                        dict_res_data['total_token_count'] = dict_res_data['prompt_token_count'] + dict_res_data['candidates_token_count']

                    str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Claude', input_mode, str_brand)
                    if not os.path.exists(str_output_dir):
                        os.makedirs(str_output_dir)                       
                    
                    str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                    with open(str_output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(dict_res_data, f, indent=4)
                    
                    time.sleep(random.randint(1, 3))
                except anthropic.APIConnectionError as e:
                    print("The server could not be reached")
                    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
                    continue
                except anthropic.InternalServerError as e:
                    time.sleep(600)
                    continue
                except anthropic.RateLimitError as e:
                    print("A 429 status code was received; we should back off a bit.")
                    continue
                except anthropic.APIStatusError as e:
                    if e.status_code == 529:
                        print('Claude Service Overloaded')
                        time.sleep(10)
                        continue 
                    elif e.status_code == 413:            
                        dict_res_data = format_model_response(str_hash, 'Request Size Exceeding Error', True, False)                        
                        str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Claude', input_mode, str_brand)
                        if not os.path.exists(str_output_dir):
                            os.makedirs(str_output_dir)
                        
                        str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                        with open(str_output_file_path, 'w', encoding='utf-8') as f:
                            json.dump(dict_res_data, f, indent=4)                        
                        continue           
                    elif e.status_code == 400:
                        if 'image exceeds' in str(e) or 'image dimensions exceed' in str(e):
                            dict_res_data = format_model_response(str_hash, 'Image Size Exceeding Error', True, False)
                            str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Claude', input_mode, str_brand)
                            if not os.path.exists(str_output_dir):
                                os.makedirs(str_output_dir)                       
                            
                            str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                            with open(str_output_file_path, 'w', encoding='utf-8') as f:
                                json.dump(dict_res_data, f, indent=4)
                            continue
                    else:
                        raise     
                    raise
                except anthropic.BadRequestError as e:
                    raise
        return
    

    def phase2_phishing_classification(self, input_dataset:InputDataset):
        str_dataset = input_dataset
        # No need to Load Phase2 Prompt. It is already loaded in the instance at Phase1.
        
        # Output: Summary
        str_output_summary_path = os.path.join(str_output_dir_base, str_dataset, 'Phase2_Claude', "Phase2_Res_Summary.csv")
        if not os.path.exists(str_output_summary_path):
            with open(str_output_summary_path, 'w') as f_summary:
                str_phase2_res_summary_hdr = f'Dataset,InputMode,Brand,Hash,Phase1Pred,Phase2Matched\n'
                f_summary.write(str_phase2_res_summary_hdr)

        for input_mode in InputMode:
            # Input: Came from Output of Phase 1
            str_input_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Claude', input_mode, str_brand)                 
            list_input_path = glob.glob(f'{str_input_dir}/*.json') # ./output/<dataset>/<input_mode>/<brand>/<hash>.json
            list_input_path = [x.replace('\\', '/') for x in list_input_path]
            list_input_path.sort()
            
            str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase2_Claude', input_mode, str_brand)
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

                # Claude
                list_model_prompt = self.create_brandcheck_prompt(input_mode, str_brand, str_phase1_pred)
                               
                try:
                    response = self.client.messages.create(
                        model=self.str_model,
                        max_tokens=4096,
                        messages=list_model_prompt
                    )

                    if not response.content:
                        # Error                
                        dict_res_data = format_phase2_response('Error', False, True)
                    else:
                        # Normal
                        str_res = response.content[0].text
                        dict_res_data = format_phase2_response(str_res)

                    if response.usage:            
                        dict_res_data['prompt_token_count'] = response.usage.input_tokens
                        dict_res_data['candidates_token_count'] = response.usage.output_tokens
                        dict_res_data['total_token_count'] = dict_res_data['prompt_token_count'] + dict_res_data['candidates_token_count']
              
                    str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                    with open(str_output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(dict_res_data, f, indent=4)

                    str_phase2_res_summary = f'{str_dataset},{input_mode},{str_brand},{str_phase1_pred},{dict_res_data['BrandMatched']}\n'
                    with open(str_output_summary_path, 'a') as f_summary:
                        f_summary.write(str_phase2_res_summary)
                    
                    time.sleep(random.randint(1, 3))
                except anthropic.APIConnectionError as e:
                    print("The server could not be reached")
                    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
                    continue
                except anthropic.InternalServerError as e:
                    time.sleep(600)
                    continue
                except anthropic.RateLimitError as e:
                    print("A 429 status code was received; we should back off a bit.")
                    continue
                except anthropic.APIStatusError as e:
                    if e.status_code == 529:
                        print('Claude Service Overloaded')
                        time.sleep(10)
                        continue 
                    elif e.status_code == 413:            
                        dict_res_data = format_phase2_response('Request Size Exceeding Error', True, False)                        
                        str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                        with open(str_output_file_path, 'w', encoding='utf-8') as f:
                            json.dump(dict_res_data, f, indent=4)                        
                        continue           
                    elif e.status_code == 400:
                        if 'image exceeds' in str(e) or 'image dimensions exceed' in str(e):
                            dict_res_data = format_phase2_response('Image Size Exceeding Error', True, False)                            
                            str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                            with open(str_output_file_path, 'w', encoding='utf-8') as f:
                                json.dump(dict_res_data, f, indent=4)
                            continue
                    else:
                        raise     
                    raise
                except anthropic.BadRequestError as e:
                    raise
        return
    

