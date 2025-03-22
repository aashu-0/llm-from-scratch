import torch
import tiktoken
from GPT import generate_text_simple

# function to text to token_id and token_ids_to_text
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special= {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    return decoded_text


# cross entropy loss for a given batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

# computing loss over all the batches in a data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) ==0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i< num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches #avg loss

# prints train and val losses after each model update
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device,
                                      num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device,
                                    num_batches=eval_iter)

        model.train()
        return train_loss, val_loss

# generate and print sample text
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(model, idx=encoded, max_new_tokens=50,
                                         context_size=context_size)

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace('\n', ' '))
    model.train()

# TRAINING LOOP
def train_model_simple(model,train_dataloader, val_dataloader, optimizer,
                       device, num_epochs, eval_freq, eval_iter, start_context,
                       tokenizer):
    train_losses, val_losses, track_tokens_seen = [],[],[]
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step +=1

            # evaluation step
            if global_step % eval_freq ==0:
                train_loss, val_loss = evaluate_model(model, train_dataloader,
                                                      val_dataloader, device,
                                                      eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)
                print(f'Epoch: {epoch+1} | Step: {global_step:06d}')
                print(f'Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')

        # prints a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

# Plot Training and Validation losses 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize =(5,3))
    ax1.plot(epochs_seen, train_losses, label= 'Training Loss')
    ax1.plot(epochs_seen, val_losses, linestyle ='-.', label = 'Validation Loss')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc= 'upper right')

    ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
    ax2 = ax1.twiny() # creates a twin x-axis (shared y-axis) for existing axis ax1

    ax2.plot(tokens_seen, train_losses, alpha=0) # alpha = 0 -> invisible plot
    ax2.set_xlabel('Tokens seen')
    fig.tight_layout()
    plt.show()