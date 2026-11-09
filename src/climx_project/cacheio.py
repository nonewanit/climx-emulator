"""Cache I/O helpers shared by the full-res scripts."""


def zarr_fp16_encoding() -> dict:
    """float16 + zstd-shuffle encoding for xarray to_zarr, compatible with
    zarr 2 (numcodecs Blosc via `compressor`) and zarr 3 (`compressors`
    list with zarr.codecs.BloscCodec). Kaggle images ship zarr 3, which
    rejects the numcodecs object outright.
    """
    import zarr

    if int(zarr.__version__.split(".")[0]) >= 3:
        from zarr.codecs import BloscCodec
        return {"dtype": "float16",
                "compressors": [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")]}
    from numcodecs import Blosc
    return {"dtype": "float16",
            "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)}
