import torch
from tqdm.auto import tqdm

def evaluate_depth(dataloader, model, device="cuda", eps=1e-6):
    """
    dataloader: DataLoader que retorna (imagen, depth_gt)
    model: red que predice mapas de profundidad
    device: "cuda" o "cpu"
    """

    rmse_list, absrel_list, d1_list, d2_list, d3_list = [], [], [], [], []

    model.eval()
    #print("hola")
    i=0
    with torch.no_grad():
        for images, depths in tqdm(dataloader, desc="Evaluando", leave=False):
            #print(f"Procesando lote {i}")
            # i += 1
            images = images.to(device)
            # images = images * (1/255.0)
            # images = (images-0.5)*2
            depths = depths.to(device)

            depths= depths.squeeze(1)
            depths = depths*(10.0 - 0.7)+0.7
            #depths = torch.clip(depths, 0.0, 1.0)
            # forward
            preds = model(images)

            # asegurar misma forma
            preds = preds.squeeze(1)   # (N,1,H,W) -> (N,H,W)
            depths = depths.squeeze(1) # idem GT
            preds= preds*(10.0 - 0.7)+0.7
            # flatten
            preds = preds.view(-1)
            gts   = depths.view(-1)

            # aplicar máscara
            mask = gts > eps
            preds = preds[mask]
            gts   = gts[mask]

            # métricas
            rmse = torch.sqrt(torch.mean((preds - gts) ** 2))
            abs_rel = torch.mean(torch.abs(preds - gts) / gts)
            ratio = torch.max(preds / gts, gts / preds)
            delta1 = torch.mean((ratio < 1.25).float())
            delta2 = torch.mean((ratio < 1.25 ** 2).float())
            delta3 = torch.mean((ratio < 1.25 ** 3).float())

            rmse_list.append(rmse.item())
            absrel_list.append(abs_rel.item())
            d1_list.append(delta1.item())
            d2_list.append(delta2.item())
            d3_list.append(delta3.item())

    # promedios globales
    results = {
        "RMSE": sum(rmse_list)/len(rmse_list),
        "AbsRel": sum(absrel_list)/len(absrel_list),
        "δ<1.25": sum(d1_list)/len(d1_list),
        "δ<1.25²": sum(d2_list)/len(d2_list),
        "δ<1.25³": sum(d3_list)/len(d3_list)
    }
    return results

# Ejemplo de uso
# preds = torch.from_numpy(np.array([...]))  # salida del DPU convertida a torch
# gts   = torch.from_numpy(np.array([...]))  # ground truth
# results = depth_metrics_torch(preds, gts)
# for k, v in results.items():
#     print(f"{k}: {v:.4f}")



def evaluate_depth_by_ranges(dataloader, model, device="cuda", eps=1e-6, bins=[(0,2),(2,5),(5,10),(10,80)]):
    """
    Evalúa métricas de profundidad por rangos de distancia
    dataloader: retorna (image, depth_gt)
    model: red que predice depth
    bins: lista de tuplas con los rangos de profundidad (en metros)
    """

    # acumuladores: dict por rango
    stats = {str(b): {"rmse": [], "absrel": [], "d1": [], "d2": [], "d3": []} for b in bins}

    model.eval()
    with torch.no_grad():
        for images, depths in tqdm(dataloader, desc="Evaluando por rangos", leave=False):
            images = images.to(device)
            depths = depths.to(device)
            depths = depths*(10.0 - 0.7)+0.7
            # forward
            preds = model(images)
            preds= preds*(10.0 - 0.7)+0.7
            preds = preds.squeeze(1)   # (N,1,H,W) → (N,H,W)
            depths = depths.squeeze(1) # (N,1,H,W) → (N,H,W)

            for (low, high) in bins:
                mask = (depths > low) & (depths <= high)

                # asegurar que hay píxeles válidos
                if mask.sum() < 1:
                    continue

                p = preds[mask]
                g = depths[mask]

                rmse = torch.sqrt(torch.mean((p - g) ** 2))
                abs_rel = torch.mean(torch.abs(p - g) / g)
                ratio = torch.max(p / g, g / p)
                delta1 = torch.mean((ratio < 1.25).float())
                delta2 = torch.mean((ratio < 1.25 ** 2).float())
                delta3 = torch.mean((ratio < 1.25 ** 3).float())

                stats[str((low,high))]["rmse"].append(rmse.item())
                stats[str((low,high))]["absrel"].append(abs_rel.item())
                stats[str((low,high))]["d1"].append(delta1.item())
                stats[str((low,high))]["d2"].append(delta2.item())
                stats[str((low,high))]["d3"].append(delta3.item())

    # promediar cada rango
    results = {}
    for b in bins:
        key = str(b)
        if len(stats[key]["rmse"]) == 0:
            continue
        results[key] = {
            "RMSE": sum(stats[key]["rmse"])/len(stats[key]["rmse"]),
            "AbsRel": sum(stats[key]["absrel"])/len(stats[key]["absrel"]),
            "δ<1.25": sum(stats[key]["d1"])/len(stats[key]["d1"]),
            "δ<1.25²": sum(stats[key]["d2"])/len(stats[key]["d2"]),
            "δ<1.25³": sum(stats[key]["d3"])/len(stats[key]["d3"])
        }
    return results
