"""Never accept a partly loaded actor with random backbone/projection parameters."""


def load_checkpoint_checked(model, state_dict, strict=False, allow_new_value_head=True):
    result = model.load_state_dict(state_dict, strict=False)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    unexpected = list(result.unexpected_keys)
    missing = []
    for key in result.missing_keys:
        if allow_new_value_head and key.startswith("value_head."):
            continue
        parameter = parameters.get(key)
        tied_loaded = parameter is not None and any(
            other in state_dict and parameters[other] is parameter for other in parameters if other != key)
        if not tied_loaded:
            missing.append(key)
    if missing or unexpected:
        raise ValueError(f"Incomplete actor checkpoint: missing={missing}, unexpected={unexpected}")
    return result
