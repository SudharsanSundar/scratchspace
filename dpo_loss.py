def per_instance_dpo_loss(lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected):
    # Format the prompt in the Alpaca template
    prompt_chosen = template.format(instruction=prompt, response=response_chosen)
    prompt_rejected = template.format(instruction=prompt, response=response_rejected)
    
    # Tokenize the prompt, making sure to add EOS token
    prompt_chosenT = tokenizer.encode(prompt_chosen + tokenizer.eos_token, return_tensors='pt')[0]
    prompt_rejectedT = tokenizer.encode(prompt_rejected + tokenizer.eos_token, return_tensors='pt')[0]

    
    output = lm_ref.forward(prompt_chosenT).logits      # Get logits (i.e. log(probs)) over vocab for the whole prompt
    denom_win = torch.sum(torch.tensor([output[i, idx] for i, idx in enumerate(prompt_chosenT)]))       # Select just the log(probs) of the desired tokens, 
                                                                                                        # then sum them to get total log prob of desired generation.
                                                                                                        # Using the trick that log(prod(probs)) = sum(log(probs)), where prod(probs) represents the unconditional probability of generating the prompt = p(x1)*p(x2 | x1)*...*p(xn | x1:n-1)

    # Same as above
    output = lm_ref.forward(prompt_rejectedT).logits
    denom_lose = torch.sum(torch.tensor([output[i, idx] for i, idx in enumerate(prompt_rejectedT)]))

    output = lm.forward(prompt_chosenT).logits
    numer_win = torch.sum(torch.tensor([output[i, idx] for i, idx in enumerate(prompt_chosenT)]))

    output = lm.forward(prompt_rejectedT).logits
    numer_lose = torch.sum(torch.tensor([output[i, idx] for i, idx in enumerate(prompt_rejectedT)]))

    # Apply the fact that log(A/B) = log(A) - log(B), along with the condition probs -> unconditional probs trick.
    dpo_loss = -(F.logsigmoid(beta * (numer_win - numer_lose + denom_lose - denom_win))) 
    dpo_loss = dpo_loss.to(lm.device)

    return dpo_loss
