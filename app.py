import streamlit as st
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from models.GTM import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset
from sklearn.metrics import mean_absolute_error

def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)
    return round(mae, 3), round(wape, 3)

def load_model(args):
    # Load category and color encodings
    data_folder= 'VISUELLE/'
    cat_dict = torch.load(os.path.join(args.data_folder, 'category_labels.pt'))
    col_dict = torch.load(os.path.join(args.data_folder, 'color_labels.pt'))
    fab_dict = torch.load(os.path.join(args.data_folder, 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(os.path.join(args.data_folder, 'gtrends.csv'), index_col=[0], parse_dates=True)

    # Create model
    if args.model_type == 'FCN':
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_trends=args.use_trends,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num
        )
    else:
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num
        )
    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)
    return model

def forecast(model, img_path, args):
    # Load and preprocess the image
    img_transforms = Compose([Resize((10, 10)), ToTensor(), Normalize(mean=[0.012, 0.010, 0.008], std=[0.029, 0.024, 0.025])])
    img = Image.open(img_path).convert('RGB')
    img = img_transforms(img).unsqueeze(0)

    # Get the product information
    category = torch.tensor([args.cat_dict['category']])
    color = torch.tensor([args.col_dict['color']])
    fabric = torch.tensor([args.fab_dict['fabric']])
    temporal_features = torch.zeros(1, 4)
    gtrends = torch.zeros(1, args.num_trends, args.trend_len)

    # Forward pass
    with torch.no_grad():
        y_pred, _ = model(category, color, fabric, temporal_features, gtrends, img)

    return y_pred.detach().cpu().numpy().flatten()[:args.output_dim]

def main():
    st.title("Zero-Shot Sales Forecasting")

    # Load model and configuration
    args = {
        'data_folder': 'VISUELLE/',
        'ckpt_path': 'log/GTM/GTM_experiment2---epoch=29---16-05-2024-08-49-43.ckpt',
        'gpu_num': 0,
        'seed': 21,
        'model_type': 'GTM',
        'use_trends': 1,
        'use_img': 1,
        'use_text': 1,
        'trend_len': 52,
        'num_trends': 3,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'output_dim': 12,
        'use_encoder_mask': 1,
        'autoregressive': 0,
        'num_attn_heads': 4,
        'num_hidden_layers': 1,
        'wandb_run': 'experiment2'
    }

    model = load_model(args)

    # Streamlit app
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_path = os.path.join('uploads', uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        forecasts = forecast(model, img_path, args)
        rescale_vals = np.load(os.path.join(args.data_folder, 'normalization_scale.npy'))
        rescaled_forecasts = forecasts * rescale_vals

        st.header("Forecasts")
        forecasts_df = pd.DataFrame(rescaled_forecasts, columns=[f"Week {i+1}" for i in range(args.output_dim)])
        st.table(forecasts_df)

        st.header("Error Metrics")
        mae, wape = cal_error_metrics(np.ones(args.output_dim) * rescaled_forecasts.mean(), rescaled_forecasts)
        st.write(f"MAE: {mae}")
        st.write(f"WAPE: {wape}")

        st.header("Forecast Plot")
        st.line_chart(forecasts_df)

if __name__ == '__main__':
    main()
