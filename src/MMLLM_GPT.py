import json
import glob
from tqdm import tqdm
import os
from MMLLM_Common import *
from openai import OpenAI

class MMLLM_GPT:
    def __init__(self, str_api_key:str):
        self.str_api_key = str_api_key
        self.dict_phase1_system_msg:dict        
        self.dict_phase1_res_format:dict
        self.dict_phase2_system_msg:dict
        self.client:OpenAI        
        
        self.str_model = "gpt-4-turbo-2024-04-09"
        self.client = OpenAI(api_key = self.str_api_key)
        return    

    def load_prompt_text(self, input_mode:InputMode):        
        # Prompts
        str_phase1_prompt_path = dict_system_prompt_path.get(input_mode)
        assert str_phase1_prompt_path is not None, f"Unknown Input mode {input_mode}"

        with open(str_phase1_prompt_path, encoding='utf-8') as f_read:
            str_phase1_system_prompt = f_read.read()        
            self.dict_phase1_system_msg = { 
                "role": "system",
                "content": [ 
                    { "type": "text", "text": str_phase1_system_prompt }, 
                    ], }

        # Response        
        str_phase1_response_prompt_path = dict_response_prompt_path.get(input_mode)
        assert str_phase1_response_prompt_path is not None, f"Unknown Input mode {input_mode}"

        with open(str_phase1_response_prompt_path, encoding='utf-8') as f_read:
            str_res_format = f_read.read()
            self.dict_phase1_res_format = {
                "role": "user",
                "content": [
                    {"type": "text", "text": str_res_format},
                ],
            }

        # Phase 2 Prompt
        str_phase2_system_prompt_path = dict_response_prompt_path.get(Phase2Mode.Phase2)
        assert str_phase2_system_prompt_path is not None, f"Unknown Input mode {Phase2Mode.Phase2}"

        with open(str_phase2_system_prompt_path, encoding='utf-8') as f_read:
            str_phase2_system_prompt = f_read.read()
            self.dict_phase2_system_msg = {
                "role": "system",
                "content": [
                    {"type": "text", "text": str_phase2_system_prompt},
                ],
            }
        return

    

    def create_identification_prompt(self, input_mode:InputMode, encoded_image, html_content):                
        '''
        First phase: Prompt for Brand identification. GPT format.
        '''
        
        dict_data_prompt = {}
        dict_data_prompt["role"] = "user"
        dict_data_prompt["content"] = []

        if input_mode == InputMode.SS:
            # Add Image
            dict_data_prompt["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    })
        elif input_mode == InputMode.HTML:
            dict_data_prompt["content"].append({
                        "type": "text",
                        "text": f"Here is the html information: {html_content}"})
        elif input_mode == InputMode.BOTH:
            dict_data_prompt["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    })
            dict_data_prompt["content"].append({
                        "type": "text",
                        "text": f"Here is the html information: {html_content}"})
        
        messages = [self.dict_phase1_system_msg, dict_data_prompt, self.dict_phase1_res_format]        
        
        return messages

    def create_brandcheck_prompt(self, str_groundtruth:str, str_prediction:str):                
        '''
        Second phase: Prompt for Brand comparison and Phishing classification. GPT format.
        '''
      
        dict_data_prompt = {}
        dict_data_prompt["role"] = "user"
        dict_data_prompt["content"] = []
     
        dict_data_prompt["content"].append({
                    "type": "text",
                    "text": f"Ground Truth: \"{str_groundtruth}\"\n\"Prediction:\"{str_prediction}\""})
        
        messages = [self.dict_phase2_system_msg, dict_data_prompt]
        
        return messages
    
    def query(self, list_messages):
        return self.client.chat.completions.create(
                    model=self.str_model,
                    messages=list_messages,
                    max_tokens=4096,
                )
        
    def phase1_brand_identification(self, input_dataset:InputDataset):
        str_dataset = input_dataset
        list_data_dir = glob.glob(f'{str_input_dir_base}/{str_dataset}/*/*/') # ../input/<dataset>/<brand>/<hash>/
        list_data_dir.sort()

        for str_data_dir in tqdm(list_data_dir, desc=f'{str_dataset}'):                             
            str_data_dir = str_data_dir.replace('\\', '/')
            list_prop = str_data_dir.split('/')
            str_ss_path = str_data_dir + '/screenshot_aft.png'
            str_html_path = str_data_dir + '/add_info.json'

            if not os.path.exists(str_ss_path) or not os.path.exists(str_html_path):
                # Error sample
                continue
            
            str_hash = list_prop[-2]
            str_brand = list_prop[-3]

            # HTML Input
            with open(str_html_path) as f_read:            
                str_html_info = json.load(f_read)['html_brand_info']     
                
            # Image Input
            encoded_image = crop_encode_image_base64(str_ss_path)

            for input_mode in InputMode:                
                self.load_prompt_text(input_mode) # Following input_mode                     

                # GPT
                list_model_prompt = self.create_identification_prompt(input_mode, encoded_image, str_html_info)
                            
                # GPT-Chat mode Query
                try:
                    response = self.query(list_model_prompt)
                    res_content = response.choices[0].message.content                    
                except Exception as e:
                    dict_res_data = format_model_response(str_hash, res_content, is_error=True, is_safety_triggered=False)
                    pass                
    
                try:
                    dict_res_data = format_model_response(str_hash, res_content, is_error=False, is_safety_triggered=False)
                except:
                    # TODO: ADD Security Message detection for GPT.
                    dict_res_data = format_model_response(str_hash, res_content, is_error=True, is_safety_triggered=False)
                    pass
                
                if response.usage:
                    dict_res_data['completion_tokens'] = response.usage.completion_tokens
                    dict_res_data['prompt_tokens'] = response.usage.prompt_tokens
                    dict_res_data['total_tokens'] = response.usage.total_tokens
            
                str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_GPT', input_mode, str_brand)
                if not os.path.exists(str_output_dir):
                    os.makedirs(str_output_dir)                       
                
                str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                with open(str_output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(dict_res_data, f, indent=4)
        return
    

    def phase2_phishing_classification(self, input_dataset:InputDataset):
        str_dataset = input_dataset
        # No need to Load Phase2 Prompt. It is already loaded in the instance at Phase1.
        
        # Output: Summary
        str_output_summary_path = os.path.join(str_output_dir_base, str_dataset, 'Phase2_GPT', "Phase2_Res_Summary.csv")
        if not os.path.exists(str_output_summary_path):
            with open(str_output_summary_path, 'w') as f_summary:
                str_phase2_res_summary_hdr = f'Dataset,InputMode,Brand,Hash,Phase1Pred,Phase2Matched\n'
                f_summary.write(str_phase2_res_summary_hdr)

        for input_mode in InputMode:
            # Input: Came from Output of Phase 1
            str_input_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_GPT', input_mode, str_brand)                 
            list_input_path = glob.glob(f'{str_input_dir}/*.json') # ./output/<dataset>/<input_mode>/<brand>/<hash>.json
            list_input_path = [x.replace('\\', '/') for x in list_input_path]
            list_input_path.sort()

            
            str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase2_GPT', input_mode, str_brand)
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

                # GPT
                list_model_prompt = self.create_brandcheck_prompt(input_mode, str_brand, str_phase1_pred)
                                
                # GPT-Chat mode Query
                try:
                    response = self.query(list_model_prompt)
                    res_content = response.choices[0].message.content                    
                    dict_res_data = format_phase2_response(res_content)
                except Exception as e:
                    dict_res_data = format_phase2_response(res_content, is_error=True, is_safety_triggered=False)
                    pass
                
                if response.usage:
                    dict_res_data['completion_tokens'] = response.usage.completion_tokens
                    dict_res_data['prompt_tokens'] = response.usage.prompt_tokens
                    dict_res_data['total_tokens'] = response.usage.total_tokens
                
                str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")                    
                with open(str_output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(dict_res_data, f, indent=4)

                str_phase2_res_summary = f'{str_dataset},{input_mode},{str_brand},{str_phase1_pred},{dict_res_data['BrandMatched']}\n'
                with open(str_output_summary_path, 'a') as f_summary:
                    f_summary.write(str_phase2_res_summary)
        return
    

