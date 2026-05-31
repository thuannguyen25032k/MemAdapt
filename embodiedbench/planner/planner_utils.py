import os
import re
import json
import base64
import copy
from mimetypes import guess_type
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
import typing_extensions as typing
from pydantic import BaseModel, Field
from json_repair import repair_json

template_lang = '''\
The output JSON format should be {"reasoning_and_reflection":str, "language_plan":str, "executable_plan":[{"action_id":int, "action_name":str}, ...]}
The fields in above JSON follow the purpose below:
1. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \
2. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \
3. executable_plan is a list of actions needed to achieve the user instruction. For each action, action_id MUST be the integer shown in parentheses next to the object name in the action table above, e.g. if the table shows "Mug(97)" under PICK UP, use action_id 97 and action_name "pick up the Mug".
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!
'''

template = '''
The output JSON format should be {"visual_state_description":str, "reasoning_and_reflection":str, "language_plan":str, "executable_plan":[{"action_id":int, "action_name":str}, ...]}
The fields in above JSON follow the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, 
4. executable_plan is a list of actions needed to achieve the user instruction. For each action, action_id MUST be the integer shown in parentheses next to the object name in the action table above, e.g. if the table shows "Mug(97)" under PICK UP, use action_id 97 and action_name "pick up the Mug".
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_lang_manip = '''\
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':str}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection: Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available. 
2. language_plan: A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name. 
3. executable_plan: A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_manip = '''\
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':str}
The fields in above JSON follows the purpose below:
1. visual_state_description: Describe the color and shape of each object in the detection box in the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input. 
2. reasoning_and_reflection: Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available. 
3. language_plan: A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name. 
4. executable_plan: A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

def fix_json(json_str):
    """
    Attempt to repair common model-output JSON errors so that json.loads() succeeds.

    Strategy (in order):
    1. Strip markdown code fences.
    2. Extract the outermost {...} block (ignore any preamble/postamble text).
    3. Fix invalid JSON escape sequences (e.g. \\find, \\navigate, \\the → remove backslash).
    4. Replace Python literals: True→true, False→false, None→null.
    5. Insert missing commas between adjacent objects/arrays (}{  →  },{  and ][  →  ],[).
    6. Remove trailing commas before ] or }.
    7. Replace single-quoted JSON structural delimiters (keys and string values)
       with double quotes, WITHOUT touching apostrophes inside already-double-quoted values.
    8. For each known long free-text field (visual_state_description,
       reasoning_and_reflection, language_plan) escape any bare double-quotes that
       appear inside the value, using a lookahead to the next JSON key boundary.
    """
    # 1. Strip markdown code fences
    json_str = json_str.replace('```json', '').replace('```', '').strip()

    # 2. Extract the outermost JSON object (handles preamble like "Output: {...}")
    brace_start = json_str.find('{')
    brace_end   = json_str.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_str = json_str[brace_start:brace_end + 1]

    # 3. Fix invalid and misused JSON escape sequences.
    #    Models often write prose backslashes like \find, \navigate, \the, \to.
    #    (a) Completely invalid escape chars (not in JSON spec at all): remove backslash.
    #        Valid JSON escapes after \: " \ / b f n r t u
    json_str = re.sub(r'\\(?!["\\/bfnrtu])', '', json_str)
    #    (b) Technically-valid escape chars used as word-starters (\find→\f+ind,
    #        \the→\t+he, \navigate→\n+avigate, \before→\b+efore, \result→\r+esult):
    #        detected when the escape char is immediately followed by another letter.
    json_str = re.sub(r'\\([bfnrt])(?=[a-zA-Z])', r'\1', json_str)
    #    (c) \u not followed by exactly 4 hex digits (e.g. \use, \under) → remove backslash.
    json_str = re.sub(r'\\u(?![0-9a-fA-F]{4})', 'u', json_str)

    # 4. Replace Python literals with JSON equivalents.
    #    Use \b word boundaries so we don't corrupt words like "Trueblood" or "FalseStart".
    json_str = re.sub(r'\bTrue\b',  'true',  json_str)
    json_str = re.sub(r'\bFalse\b', 'false', json_str)
    json_str = re.sub(r'\bNone\b',  'null',  json_str)

    # 5. Insert missing commas between adjacent } { or ] [ pairs.
    #    This fixes executable_plan items like [...}{...] that InternVL sometimes emits.
    json_str = re.sub(r'}\s*{', '},{', json_str)
    json_str = re.sub(r']\s*\[', '],[', json_str)

    # 6. Remove trailing commas before a closing ] or }.
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    # 7. Convert single-quoted keys/string-delimiters to double quotes.
    #    Only replace ' that act as JSON delimiters (immediately after { , [ or :),
    #    to avoid turning apostrophes inside already-double-quoted string values.

    # Targeted replacement of single-quoted keys: {'key': ...}
    json_str = re.sub(r"(?<=[{,])\s*'([^']+)'\s*:", lambda m: f' "{m.group(1)}":', json_str)

    # Replace remaining single-quote string delimiters that wrap values
    # (preceded by : or [ or , followed by optional space)
    json_str = re.sub(r"(?<=[:,\[])\s*'((?:[^'\\]|\\.)*)'", lambda m: f' "{m.group(1)}"', json_str)

    # 8. Escape bare double-quotes inside free-text string values.
    #    For each long-text field, find the value between its opening quote and the
    #    quote that precedes the next key (or end of object).
    text_fields = [
        'visual_state_description',
        'reasoning_and_reflection',
        'language_plan',
    ]
    next_key_pattern = r'(?="\s*,\s*"(?:' + '|'.join(text_fields + ['executable_plan']) + r')"|\s*"\s*})'

    for field in text_fields:
        pattern = r'("' + field + r'"\s*:\s*")(?P<value>.*?)' + next_key_pattern
        def _escape_inner(match):
            prefix = match.group(1)
            value  = match.group('value')
            # Escape any double-quote not already escaped
            value  = re.sub(r'(?<!\\)"', r'\\"', value)
            return prefix + value
        json_str = re.sub(pattern, _escape_inner, json_str, flags=re.DOTALL)

    # 9. Last-resort: if the result is still not valid JSON, delegate to json-repair.
    try:
        json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        json_str = repair_json(json_str, return_objects=False)

    return json_str


class ExecutableAction_1(typing.TypedDict): 
    action_id: int = Field(
        description="The action ID to select from the available actions given by the prompt"
    )
    action_name: str = Field(
        description="The name of the action"
    )
class ActionPlan_1(BaseModel):
    visual_state_description: str = Field(
        description="Description of current state from the visual image"
    )
    reasoning_and_reflection: str = Field(
        description="summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task"
    )
    language_plan: str = Field(
        description="The list of actions to achieve the user instruction. Each action is started by the step number and the action name"
    )
    executable_plan: list[ExecutableAction_1] = Field(
        description="A list of actions needed to achieve the user instruction, with each action having an action ID and a name."
    )

class ActionPlan_1_manip(BaseModel):
    visual_state_description: str = Field(
        description="Describe the color and shape of each object in the detection box in the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input."
    )
    reasoning_and_reflection: str = Field(
        description="Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available."
    )
    language_plan: str = Field(
        description="A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name."
    )
    executable_plan: str = Field(
        description="A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action."
    )

def convert_format_2claude(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":
            new_content = []
    
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages

def convert_format_2gemini(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":

            new_content = []
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_data}"
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)
        
    return new_messages



class ExecutableAction(typing.TypedDict): 
    action_id: int
    action_name: str
class ActionPlan(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction]

class ActionPlan_manip(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: str

class ExecutableAction_lang(typing.TypedDict): 
    action_id: int
    action_name: str
class ActionPlan_lang(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction_lang]

class ActionPlan_lang_manip(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: str

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def truncate_message_prompts(message_history: list):
    """
    Traverse the message list and truncate the part before "------------" in the text content of all messages except the last one
    
    Args:
        message_history: Message list, each message contains role and content
        
    Returns:
        list: Processed message list
    """
    if not message_history:
        return message_history
        
    # Create a deep copy to avoid modifying the original data
    processed_messages = []
    
    # Process all messages except the last one
    for i, message in enumerate(message_history):
        if i == len(message_history) - 1:
            # Keep the last message unchanged
            processed_messages.append(message)
        else:
            # Process current message
            processed_message = {
                "role": message.get("role", ""),
                "content": []
            }
            
            # Traverse content list
            for content_item in message.get("content", []):
                if content_item.get("type") == "text":
                    # Process text type content
                    text_content = content_item.get("text", "")
                    
                    # Look for "----------" separator
                    if "----------" in text_content:
                        # Truncate content before separator, keep content after separator
                        truncated_text = text_content.split("----------")[1]
                    else:
                        # If no separator found, keep original text
                        truncated_text = text_content
                        
                    processed_content_item = content_item.copy()
                    processed_content_item["text"] = truncated_text
                    processed_message["content"].append(processed_content_item)
                else:
                    # Directly copy non-text type content
                    processed_message["content"].append(content_item.copy())
            
            processed_messages.append(processed_message)
    
    return processed_messages