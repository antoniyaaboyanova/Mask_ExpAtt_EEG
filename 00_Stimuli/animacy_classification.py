from classification_utils import *
import argparse
# 1. Load neccesary data
data = load_data("/projects/crunchie/boyanova/EEG_Things/MEx_EEG/Animacy/llama_animacy.pkl")
df = pd.DataFrame(data)

parser = argparse.ArgumentParser(description="Animacy classification preprocessing")
parser.add_argument(
    "--mooney",
    action="store_true",
    help="Use Mooney images instead of natural images"
)


parser.add_argument(
    "--preprocess",
    choices=["crop", "resize"],
    default="resize",
    help="Image preprocessing: center crop or resize to 224x224"
)

args = parser.parse_args()

mooney = args.mooney
preprocess_mode = args.preprocess

if mooney:
    df["path"] = df["path"].str.replace(
        "/Image_dataset/Images/",
        "/Image_dataset/Mooney_Images/",
        regex=False
    )
    folder_name = "mooney"
else:
    folder_name = "natural"
    
print(df.path[0])

# 2. Generate paths
animacy_paths = []
inanimacy_paths = []

for idx in tqdm(range(len(df))):
    data = df.iloc[idx]
    files = os.listdir(data.path)
    file_paths = [os.path.join(data.path, f) for f in files]
    if data.animate_llama == "animate":
        animacy_paths.extend(file_paths)
    else:
        inanimacy_paths.extend(file_paths)

# 3. Initiate results dict + constant variables
all_image_paths = animacy_paths + inanimacy_paths
animacy_labels = (
    [True] * len(animacy_paths) +
    [False] * len(inanimacy_paths)
)

results = OrderedDict()
for path, is_anim in zip(all_image_paths, animacy_labels):
    results[path] = {
        "is_animate": is_anim,
        "acc": [],
        "prob_animate": [],
        "prob_inanimate": [],
        "shuffle": [],
        "fold": []
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ANIMATE, N_INANIMATE = len(animacy_paths), len(inanimacy_paths)

N_FOLDS = 5
N_SHUFFLES = 25
EPOCHS = 50
BATCH_SIZE = 64

losses = np.empty((N_SHUFFLES, N_FOLDS, EPOCHS))
accs_train = np.empty((N_SHUFFLES, N_FOLDS, EPOCHS))



#### Transforms ####
if preprocess_mode == "crop":
    PREPROCESS = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    preprocess_folder = "cropped"
else:  # resize
    preprocess_folder = "resized"
    PREPROCESS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

SAVE_DIR = f"/projects/archiv/DataStore_Boyanova/ExpAtt_EEG/Image_dataset/Animacy_model_perf/{preprocess_folder}/{folder_name}/cornet-s_performance.pkl"
os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)
# 4. Main Loop
for shuffle in range(N_SHUFFLES):
    print(f"Shuffle: {shuffle}")
    rng = np.random.default_rng(seed=shuffle)

    core_dataset, labels, indices, extra_dataset = build_core_and_extra_datasets(
        animacy_paths=animacy_paths,
        inanimacy_paths=inanimacy_paths,
        preprocess=PREPROCESS,
        batch_size=BATCH_SIZE,
        rng=rng
    )

    skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=shuffle
    )
    # Cross-validation on core set
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):

        train_set = Subset(core_dataset, train_idx)
        test_set = Subset(core_dataset, test_idx)

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Train model
        model = cornet_two_class(device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )

        model, loss_, acc_ = train_model(
            model=model,
            dataloader_train=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=EPOCHS,
            device=DEVICE
        )

        losses[shuffle, fold, :] = loss_
        accs_train[shuffle, fold, :] = acc_

        # CV test
        test_model(
            model=model,
            dataloader=test_loader,
            device=DEVICE,
            results=results,
            shuffle_id=shuffle,
            fold_id=fold
        )

        # Save continuously
        dump_data(results, SAVE_DIR)
        loss_name = os.path.join(os.path.dirname(SAVE_DIR), "cornet-s_train_loss.npy")
        acc_name = os.path.join(os.path.dirname(SAVE_DIR), "cornet-s_train_acc.npy")
        np.save(loss_name, losses)
        np.save(acc_name, accs_train)

        # Test remaining images only once
        if fold == 0:
            test_model(
                model=model,
                dataloader=extra_dataset,
                device=DEVICE,
                results=results,
                shuffle_id=shuffle,
                fold_id=None  # marks extra set
            )

            # Save again (extra safety)
            dump_data(results, SAVE_DIR)
