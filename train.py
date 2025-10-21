import os, argparse, time, json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import YoloTxtDataset
from model import YoloLite
from loss import YoloLiteCriterion

def collate_fn(batch):
    # batch: list of (img_tensor, targets_tensor, path)
    import torch  # 放这里避免循环依赖
    imgs   = torch.stack([x[0] for x in batch], dim=0)  # [B,3,H,W]
    t_list = [x[1] for x in batch]                      # list[Tensor(Ni,5)]
    p_list = [x[2] for x in batch]                      # list[str]
    return imgs, t_list, p_list

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", type=str, required=True, help=".../train/images")
    ap.add_argument("--train_labels", type=str, required=True, help=".../train/labels")
    ap.add_argument("--val_images",   type=str, required=True, help=".../val/images")
    ap.add_argument("--val_labels",   type=str, required=True, help=".../val/labels")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--device", type=str, default="mps")  # mps/cuda/cpu
    ap.add_argument("--out", type=str, default="runs_yololite/exp")
    return ap.parse_args()

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    meters = []
    for imgs, targets, _ in loader:
        imgs = imgs.to(device).float()
        # targets 是 list[Tensor(Ni,5)] 的形式给到 loss
        t_list = [t.to(device).float() for t in targets]
        pred = model(imgs)
        loss, m = criterion(pred, t_list)
        meters.append(m)
    if len(meters) == 0:
        return {}
    keys = meters[0].keys()
    avg = {k: sum(m[k] for m in meters) / len(meters) for k in keys}
    return avg



def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    # data
    train_set = YoloTxtDataset(args.train_images, args.train_labels, args.imgsz)
    val_set   = YoloTxtDataset(args.val_images,   args.val_labels,   args.imgsz)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,  # 同上
        collate_fn=collate_fn
    )

    # model & loss & opt
    model = YoloLite(nc=args.nc).to(device)
    criterion = YoloLiteCriterion(nc=args.nc)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = 1e9
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), ncols=100, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, targets, _ in pbar:
            imgs = imgs.to(device).float()
            t_list = [t.to(device).float() for t in targets]
            optimizer.zero_grad()
            pred = model(imgs)
            loss, m = criterion(pred, t_list)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(m)

        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, **val_metrics})
        print("val:", {k: round(v, 4) for k, v in val_metrics.items()})

        # 简单“最优”判断
        cur = val_metrics.get("loss", 1e9)
        if cur < best_loss:
            best_loss = cur
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

        # 每轮都存个最新
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))

        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"Done. best_loss={best_loss:.4f}, weights saved to {args.out}")

if __name__ == "__main__":
    main()
