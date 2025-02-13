from training.train_lora import TrainConfig, train_model


config = TrainConfig(
    model_name="microsoft/deberta-v3-base",
    epochs=10,
    max_length=512,
    learning_rate=1e-4,
    batch_size=16,
    test_size=0.20
    )

tatrainer = train_model(config, ["data"], "output_dir")
