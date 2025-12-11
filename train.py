import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from tqdm import tqdm
import argparse

from encodec import EncodecModel
from encodec.utils import convert_audio
from soundstorm_pytorch import SoundStorm, ConformerWrapper


def load_encodec():
    encodec = EncodecModel.encodec_model_24khz()
    encodec.set_target_bandwidth(6.0)
    encodec.eval()
    return encodec


@torch.no_grad()
def wav_to_tokens(path, encodec, max_len=1024, num_q=8, device="cuda"):
    wav_np, sr = sf.read(os.path.abspath(path))
    
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]
    
    wav = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
    wav = convert_audio(wav, sr, 24000, 1)
    
    encoded = encodec.encode(wav)
    codes = encoded[0][0]
    
    codes_list = []
    for q in range(num_q):
        cb = codes[:, q, :].squeeze(0).long()
        T = cb.shape[0]
        
        if T < max_len:
            cb = torch.nn.functional.pad(cb, (0, max_len - T))
        else:
            cb = cb[:max_len]
        
        codes_list.append(cb)
    
    tokens = torch.stack(codes_list, dim=-1)
    return tokens.to(device)


class ESC50Dataset(Dataset):
    def __init__(self, json_path, encodec, device="cuda"):
        meta = json.load(open(os.path.abspath(json_path)))
        self.data = meta["data"]
        self.encodec = encodec
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        it = self.data[idx]
        tokens = wav_to_tokens(it["wav"], self.encodec, device=self.device)
        return tokens


def train_soundstorm(args):
    device = torch.device(args.device)
    encodec = load_encodec()
    
    train_json = f"{args.data_dir}/esc_train_data_{args.fold}.json"
    eval_json = f"{args.data_dir}/esc_eval_data_{args.fold}.json"
    
    train_dataset = ESC50Dataset(train_json, encodec, device)
    eval_dataset = ESC50Dataset(eval_json, encodec, device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )
    
    conformer = ConformerWrapper(
        codebook_size=1024,
        num_quantizers=args.num_quantizers,
        conformer=dict(
            dim=args.hidden_dim,
            depth=args.depth,
            attn_flash=False
        ),
    )
    
    model = SoundStorm(
        conformer,
        steps=args.steps,
        schedule='cosine'
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for tokens in pbar:
            tokens = tokens.to(device)
            
            loss = model(tokens, return_loss=True)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for tokens in eval_loader:
                tokens = tokens.to(device)
                loss = model(tokens, return_loss=True)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(eval_loader)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, f"{args.output_dir}/best_model_fold{args.fold}.pt")
        
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, f"{args.output_dir}/checkpoint_epoch{epoch+1}_fold{args.fold}.pt")
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_quantizers", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_soundstorm(args)
