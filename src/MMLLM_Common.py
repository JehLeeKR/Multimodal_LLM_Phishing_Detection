
import re
from enum import Enum
import PIL.Image
import base64
from io import BytesIO
import os

class Phase2Mode(Enum):
    Phase2 = 'phase2'

class InputMode(Enum):
    SS = 'ss'
    HTML = 'html'
    BOTH = 'both'    

class InputDataset(Enum):
    MMLLM_Benign = 'MMLLM_Benign' 
    MMLLM_Phishing = 'MMLLM_Phishing'
    APW_Wild = 'APW-Wild'
    Pert_BG = 'Pert-BG'
    Pert_Foot = 'Pert-BG'
    Pert_Text = 'Pert-Text'
    Pert_Typo = 'Pert-Typo'

str_input_dir_base = '../input/'
str_output_dir_base = '../output/'

dict_system_prompt_path = {
        InputMode.SS:'../prompts/system_prompt_ss.txt',
        InputMode.HTML:'../prompts/system_prompt_html.txt',
        InputMode.BOTH:'../prompts/system_prompt_both.txt',
        Phase2Mode.Phase2:'../prompts/system_prompt_phase2.txt'        
    }

dict_response_prompt_path = {
        InputMode.SS:'../prompts/response_format_prompt.txt',
        InputMode.HTML:'../prompts/response_format_prompt_html.txt',
        InputMode.BOTH:'../prompts/response_format_prompt.txt'
    }

def crop_encode_image_PIL(str_image_path:str):
    # Image Input. Load once, use in multiple modes
    # ----------------------------------------------------------------------
    # Image Crop for GeminiProv-Vision.
    # Gemini : Image Base64 Size Lim. 7MB. App. 5MB in Image.        
    int_gemini_max_img = 5*1024*1024

    int_img_file_size = os.path.getsize(str_image_path)
    if int_img_file_size > int_gemini_max_img:
        f_reduce_ratio = int_gemini_max_img/int_img_file_size

        # Crop, instead of Resize.
        image = PIL.Image.open(str_image_path)
        int_cur_width = image.width
        int_cur_height = image.height
        int_reduce_height = int(int_cur_height*f_reduce_ratio)
        image = image.crop((0, 0, int_cur_width, int_reduce_height))
    else:
        image = PIL.Image.open(str_image_path)
    # ----------------------------------------------------------------------
    return image

def crop_encode_image_base64(str_image_path:str):
    # Claude3 Image MaxHeight: 1568 
    int_max_height = 1568
    im = PIL.Image.open(str_image_path)
    int_cur_width = im.width
    int_cur_height = im.height
    if int_cur_height > int_max_height:
        im = im.crop((0, 0, int_cur_width, int_max_height))
        buffered = BytesIO()
        if '.png' in str_image_path:
            im.save(buffered, format="png")
        else:
            im.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        with open(str_image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")        
            
def search_for_response(pattern, response_text):
    try: 
        return re.search(pattern, response_text).group(1).strip()
    except:
        return ""
    
def format_model_response(folder_hash, response_text:str, is_error:bool=False, is_safety_triggered:bool=False):
    if is_error:
        b_error = True
        brand = has_credentials = has_call_to_actions = list_of_credentials = list_of_call_to_action = confidence_score = supporting_evidence = "Error Occurred"
    elif is_safety_triggered:
        b_error = True
        brand = has_credentials = has_call_to_actions = list_of_credentials = list_of_call_to_action = confidence_score = supporting_evidence = "Safety Reasons"
    elif "payload size exceeds the limit" in response_text:
        b_error = True
        brand = has_credentials = has_call_to_actions = list_of_credentials = list_of_call_to_action = confidence_score = supporting_evidence = "Payload exceeds limit"
    elif len(response_text) == 0:
        b_error = True
        brand = has_credentials = has_call_to_actions = list_of_credentials = list_of_call_to_action = confidence_score = supporting_evidence = "Indeterminate"
    else: 
        b_error = False
        brand = search_for_response(r'Brand: (.+)', response_text)
        has_credentials = search_for_response(r'Has_Credentials: (.+)', response_text)
        has_call_to_actions = search_for_response(r'Has_Call_To_Action: (.+)', response_text)
        list_of_credentials = search_for_response(r'List_of_credentials: (.+)', response_text)
        list_of_call_to_action = search_for_response(r'List_of_call_to_action: (.+)', response_text)
        confidence_score = search_for_response(r'Confidence_Score: (.+)', response_text)
        supporting_evidence = search_for_response(r'Supporting_Evidence: (.+)', response_text)
        
    return {
        "Folder Hash": folder_hash,
        "Brand": brand,
        "Has_Credentials": has_credentials,
        "Has_Call_To_Actions": has_call_to_actions,
        "List of Credentials fields": list_of_credentials,
        "List of Call-To-Actions": list_of_call_to_action,
        "Confidence Score": confidence_score,
        "Supporting Evidence": supporting_evidence,
        "Error":b_error
    }

def format_phase2_response(response_text:str, is_error:bool, is_safety_triggered:bool):
    if is_error:
        b_matched = False
        str_explanation = "Error Occurred"
        b_error = True
    elif is_safety_triggered:
        b_matched = False
        str_explanation = "Safety Reasons"
        b_error = True
    elif "payload size exceeds the limit" in response_text:
        b_matched = False
        str_explanation = "Payload exceeds limit"        
        b_error = True
    elif len(response_text) == 0:
        b_matched = False
        str_explanation = "Indeterminate"
        b_error = True
    else: 
        b_matched = search_for_response(r'BrandMatch: (.+)', response_text)
        str_explanation = search_for_response(r'Explanation: (.+)', response_text)
        b_error = False
                
    return {
        "BrandMatched": b_matched,
        "Explanation": str_explanation,
        "Error":b_error
    }
