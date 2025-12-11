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
def wav_to_tokens(path, encodec, max_len=1024, num_q=12, device="cuda"):
    wav_np, sr = sf.read(os.path.abspath(path))
    
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]
    
    wav = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
    wav = convert_audio(wav, sr, 24000, 1)
    
    encoded = encodec.encode(wav)
    cb = encoded[0][0][:, 0].squeeze(0).long()
    
    T = cb.shape[0]
    if T < max_len:
        cb = torch.nn.functional.pad(cb, (0, max_len - T))
    else:
        cb = cb[:max_len]
    
    tokens = cb.unsqueeze(-1).repeat(1, num_q)
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
        label = int(it["labels"][-2:])
        return tokens, label


class SoundStormClassifier(nn.Module):
    def __init__(self, num_quantizers=12, num_classes=50, freeze_encoder=True):
        super().__init__()
        
        self.num_quantizers = num_quantizers
        
        self.conformer = ConformerWrapper(
            codebook_size=1024,
            num_quantizers=num_quantizers,
            conformer=dict(
                dim=512,
                depth=2,
                attn_flash=False
            ),
        )
        
        if freeze_encoder:
            self.conformer.requires_grad_(False)
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        B, T, Q = x.shape
        x = x.reshape(B, T * Q)
        
        feat = self.conformer(x, return_embeddings=True)
        feat = feat.mean(dim=1)
        
        return self.classifier(feat)


def train_fold(args):
    device = torch.device(args.device)
    encodec = load_encodec()
    
    train_json = f"{args.data_dir}/esc_train_data_{args.fold}.json"
    eval_json = f"{args.data_dir}/esc_eval_data_{args.fold}.json"
    
    train_dataset = ESC50Dataset(train_json, encodec, device)
    eval_dataset = ESC50Dataset(eval_json, encodec, device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    model = SoundStormClassifier(
        num_quantizers=args.num_quantizers,
        num_classes=args.num_classes,
        freeze_encoder=args.freeze_encoder
    ).to(device)
    
    optimizer = optim.Adam(
        model.classifier.parameters() if args.freeze_encoder else model.parameters(),
        lr=args.learning_rate
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for tokens, labels in pbar:
            tokens = tokens.to(device)
            labels = labels.to(device)
            
            logits = model(tokens)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for tokens, labels in eval_loader:
                tokens = tokens.to(device)
                labels = labels.to(device)
                
                logits = model(tokens)
                preds = logits.argmax(dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        eval_acc = correct / total
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Eval Accuracy: {eval_acc:.4f}")
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(
                model.state_dict(),
                f"{args.output_dir}/best_model_fold{args.fold}.pt"
            )
    
    print(f"\nBest validation accuracy: {best_acc:.4f}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_quantizers", type=int, default=12)
    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_fold(args)
