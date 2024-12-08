def chat_batch(
    self,
    tokenizer,
    pixel_values_list,
    questions,
    generation_config,
    histories=None,
    return_histories=False,
    num_patches_lists=None,
    IMG_START_TOKEN='<|vision_start|>',
    MG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|vision_pad|>',
    verbose=False,
    visual_features_list=None
):
    """
    This function facilitates a batched chat interaction by generating responses to a list of questions. 
    It incorporates image-based tokens and histories for context and prepares the input for the generation model.

    Args:
        tokenizer (Tokenizer): Tokenizer for processing text inputs.
        pixel_values_list (List[torch.FloatTensor]): List of pixel values for images.
        questions (List[str]): List of questions for the chat model.
        generation_config (dict): Configuration for text generation.
        histories (List[List[Tuple[str, str]]], optional): Previous question-answer pairs for each conversation. Defaults to None.
        return_histories (bool, optional): Whether to return the updated histories. Defaults to False.
        num_patches_lists (List[int], optional): Number of patches for each image. Defaults to None.
        IMG_START_TOKEN (str, optional): Token marking the start of an image. Defaults to '<|vision_start|>'.
        MG_END_TOKEN (str, optional): Token marking the end of an image. Defaults to '<|vision_end|>'.
        IMG_CONTEXT_TOKEN (str, optional): Token used as context padding for image. Defaults to '<|vision_pad|>'.
        verbose (bool, optional): Whether to print queries and responses. Defaults to False.
        visual_features_list (List[torch.FloatTensor], optional): Precomputed visual features for images. Defaults to None.

    Returns:
        List[str]: Generated responses for each question.
        List[List[Tuple[str, str]]]: Updated histories (if `return_histories` is True).
        """
    if histories is None:
        histories = [[] for _ in questions]


    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_context_token_id = img_context_token_id
    # Get eos_token_id from the template
    template = get_conv_template(self.template)
    template.system_message = self.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
    generation_config['eos_token_id'] = eos_token_id

    queries = []
    input_ids_list=[]
    attention_mask_list=[]

    for idx in range(len(questions)):
        question = questions[idx]
        history = histories[idx]
        pixel_values = pixel_values_list[idx] if pixel_values_list[idx] is not None else None
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        if not history and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        template_i = get_conv_template(self.template)
        template_i.system_message = self.system_message
        for (old_question, old_answer) in history:
            template_i.append_message(template_i.roles[0], old_question)
            template_i.append_message(template_i.roles[1], old_answer)
        template_i.append_message(template_i.roles[0], question)
        template_i.append_message(template_i.roles[1], None)
        query = template_i.get_prompt()
                # Handle image tokens
        if pixel_values is not None:
            for num_patches in num_patches_list:
                tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)] + ["<tile_global_thumbnail>"]
                image_tokens = ''
                for tile_pos_identifier in tile_pos_identifiers:
                    image_tokens += tile_pos_identifier + IMG_CONTEXT_TOKEN * self.num_image_token
                image_tokens = '<Image>' + image_tokens + '</Image>'
                query = query.replace('<image>', image_tokens, 1)
        # queries.append(query)
        # Tokenize all queries together
        model_inputs = tokenizer(
            query,
            return_tensors='pt',
            padding=True,
            truncation=True  # Ensure that sequences are truncated to the model's max length
        )
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)


    # Prepare pixel_values batch

    
    # Call the generate function
    generation_output = self.generate_batch(
        pixel_values_list=pixel_values_list,
        input_ids_list=input_ids_list,
        attention_mask_list=attention_mask_list,
        **generation_config
    )
    responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    
    outputs = []
    for idx, response in enumerate(responses):
        response = response.split(template.sep)[0].strip()
        histories[idx].append((questions[idx], response))
        outputs.append(response)

    if return_histories:
        return outputs, histories
    else:
        if verbose:
            for idx, query in enumerate(queries):
                query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
                query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
                print(query_to_print, outputs[idx])
        return outputs






@torch.no_grad()
def generate_batch(
    self,
    pixel_values_list: Optional[List[torch.FloatTensor]] = None,
    input_ids_list: Optional[List[torch.FloatTensor]] = None,
    attention_mask_list: Optional[List[torch.LongTensor]] = None,
    visual_features: Optional[torch.FloatTensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """
    This function generates responses in a batched manner by processing input embeddings and attention masks.
    It optionally integrates image-related features and configurations for text generation.

    Args:
        pixel_values_list (List[torch.FloatTensor], optional): List of pixel values for images.
        input_ids_list (List[torch.FloatTensor], optional): List of tokenized input IDs.
        attention_mask_list (List[torch.LongTensor], optional): List of attention masks.
        visual_features (torch.FloatTensor, optional): Precomputed visual features for images. Defaults to None.
        generation_config (GenerationConfig, optional): Configuration for text generation. Defaults to None.
        output_hidden_states (bool, optional): Whether to return hidden states. Defaults to None.
        return_dict (bool, optional): Whether to return outputs as a dictionary. Defaults to None.
        **generate_kwargs: Additional keyword arguments for text generation.

    Returns:
        torch.LongTensor: Generated token IDs for the responses.
    """
    input_embeds_list = []
    attention_mask_padded_list = []

    # Determine the maximum sequence length
    max_seq_length = max(input_ids.shape[1] for input_ids in input_ids_list)

    # Process each input

    for pixel_values, input_ids, attention_mask in zip(pixel_values_list, input_ids_list, attention_mask_list):
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features.cuda()
                vit_embeds = self.mlp1(vit_embeds)
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0, "No valid image context token IDs found."
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Pad input_embeds and attention_mask to match max_seq_length
        seq_length = input_embeds.shape[1]
        if seq_length < max_seq_length:
            pad_size = max_seq_length - seq_length
            input_embeds = F.pad(input_embeds, (0, 0, 0, pad_size))  # Pad sequence dimension
            attention_mask = F.pad(attention_mask, (0, pad_size))    # Pad sequence dimension

        input_embeds_list.append(input_embeds)
        attention_mask_padded_list.append(attention_mask)
    

    # Concatenate inputs for batch processing
    input_embeds = torch.cat(input_embeds_list, dim=0)
    attention_mask = torch.cat(attention_mask_padded_list, dim=0)
    
    # Generate outputs
    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs