def patch_and_eval(val_fn, target, attr, new_value):
    """
    把 target.attr 临时替换为 new_value，调用 val_fn()，然后还原。

    Args:
        val_fn:    () -> metric
        target:    任意对象，比如 model.DG、model、model.branch
        attr:      属性名字符串，比如 "forward"、"branch"、"encoder"
        new_value: 替换值，nn.Module 或函数均可

    Returns:
        val_fn() 的返回值

    Examples:
        # 替换 pooling（forward 级别）
        patch_and_eval(val_fn, model.DG, "forward", my_forward)

        # 替换子模块（attribute 级别）
        patch_and_eval(val_fn, model.DG, "branch", my_branch_module)
    """
    old = getattr(target, attr)
    setattr(target, attr, new_value)
    try:
        return val_fn()
    finally:
        setattr(target, attr, old)
