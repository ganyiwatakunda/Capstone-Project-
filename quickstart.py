from pydrive.auth import GoogleAuth

drive_auth = GoogleAuth()

drive_auth.settings[‘client_config_file’] = r’example\client_secrets.json’

drive_auth.LocalWebserverAuth()

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

    # Load the model checkpoint from Google Drive
    #model.load_state_dict(torch.load(load_model_from_gdrive('1GanC5jIQVS3C9WpRkYyre6424jJ3r83w', 'MyDrive/VISUELLE/GTM_experiment2---epoch=29---16-05-2024-08-49-43.ckpt'))['state_dict'], strict=False)
    
    
    return model
    #model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)
    #return model

