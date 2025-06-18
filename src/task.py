from PIL import Image

def vsp_preprocess_function(sample):
    # Load images
    image = Image.open(sample["image_input"]).convert("RGB") 
    image_output = Image.open(sample["image_output"]).convert("RGB")

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_output},
                {"type": "text", "text": sample["text_output"]},
                ],
        }
    ]

    return conversations

task_preporcess_config = {
    'vsp-spatial-reasoning': vsp_preprocess_function,
    'vsp-spatial-planning': vsp_preprocess_function,
}