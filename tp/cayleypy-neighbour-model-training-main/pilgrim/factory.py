from .model import Pilgrim


def build_model(
    *,
    num_classes,
    state_size,
    output_dim=1,
    dropout_rate=0.0,
    hd1=1024,
    hd2=256,
    nrd=4,
):
    return Pilgrim(
        num_classes=num_classes,
        state_size=state_size,
        hd1=hd1,
        hd2=hd2,
        nrd=nrd,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    )


def build_model_from_info(
    info,
    *,
    num_classes,
    state_size,
    output_dim=1,
    all_moves=None,
    move_names=None,
):
    del all_moves, move_names
    return build_model(
        num_classes=num_classes,
        state_size=state_size,
        output_dim=output_dim,
        dropout_rate=float(info.get("dropout", 0.0)),
        hd1=int(info.get("hd1", 1024)),
        hd2=int(info.get("hd2", 256)),
        nrd=int(info.get("nrd", 4)),
    )
