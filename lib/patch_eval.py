"""
patch_eval.py

用法:
    from lib.patch_eval import patch_and_eval

    def my_pooling(enc, shape_flag):
        # enc: (B, M, F)   shape_flag: (B, M)
        # 返回: (B, 1, F)
        return (enc * shape_flag.unsqueeze(-1)).amax(1, keepdim=True)

    metric = patch_and_eval(model, val_fn, my_pooling)
"""


def patch_and_eval(model, val_fn, new_pooling_fn):
    """
    Args:
        model:          PI_GANO 实例
        val_fn:         () -> metric，你的验证函数，无参调用
        new_pooling_fn: (enc, shape_flag) -> Domain_enc
                        enc: (B,M,F), shape_flag: (B,M), 返回 (B,1,F)
    Returns:
        val_fn() 的返回值
    """
    dg = model.DG
    original_forward = dg.forward

    def patched_forward(shape_coor, shape_flag):
        enc = dg.branch(shape_coor)               # (B, M, F)
        return new_pooling_fn(enc, shape_flag)    # (B, 1, F)

    dg.forward = patched_forward
    try:
        return val_fn()
    finally:
        dg.forward = original_forward
