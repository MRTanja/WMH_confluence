import tempfile
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import your functions
from confluence_metric import calculate_confluence, calculate_volume

st.set_page_config(page_title="WMH Confluence", layout="centered")

st.title("WMH Confluence Calculator")

st.write(
    """
Upload a WMH segmentation (.nii.gz) to calculate the
confluence metric. Optionally, upload the corresponding FLAIR 
image to view it the WMH mask overlaid.
"""
)

uploaded_file = st.file_uploader(
    "WMH segmentation",
    type=["nii", "gz"]
)

flair_file = st.file_uploader(
    "Optional FLAIR image",
    type=["nii", "nii.gz"]
)

s = st.number_input(
    "Gaussian kernel width (s)",
    value=0.05,
    step=0.01
)

voxel_size = st.number_input(
    "Voxel volume (mm³)",
    value=1.0
)
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp.flush()

        img = nib.load(tmp.name).get_fdata()

    # Calculate confluence
    confluence = [
        calculate_confluence(img, i, s)
        for i in range(img.shape[2])
    ]

    volume = [
        calculate_volume(img, i, voxel_size)
        for i in range(img.shape[2])
    ]


    confluence = pd.Series(confluence)
    volume = pd.Series(volume)

    conf_norm = confluence / volume

    result = pd.DataFrame(
        {
            "volume": [volume.sum()],
            "confluence_scaled": [
                conf_norm.sum()
                / (30.20349728 * (confluence > 0).sum())
            ],
        }
    )

    st.success("Finished!")
    st.subheader(
        f"**Confluence:** {result['confluence_scaled'].iloc[0]:.4f}"
    )
    st.subheader(
        f"**Volume:** {result['volume'].iloc[0]:.1f} mm³"
    )
    st.dataframe(result)

    st.download_button(
        "Download CSV",
        result.to_csv(index=False),
        file_name="confluence.csv",
        mime="text/csv",
    )

    # Viewer
    st.subheader("Axial view")

    # Load FLAIR if provided
    flair_img = None

    if flair_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
            tmp.write(flair_file.getvalue())
            tmp.flush()

            flair_img = nib.load(tmp.name).get_fdata()

        # Handle 4D images
        if flair_img.ndim == 4:
            flair_img = flair_img[:, :, :, 0]

    if flair_img is not None and flair_img.shape != img.shape:
        st.warning(
            f"FLAIR shape {flair_img.shape} does not match WMH shape {img.shape}. "
            "Overlay may be misaligned."
    )

    # Slice selector
    slice_idx = st.slider(
        "Select slice",
        0,
        img.shape[2] - 1,
        img.shape[2] // 2
    )

    # Toggle WMH overlay
    show_wmh = st.checkbox(
        "Show WMH overlay",
        value=True
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    if flair_img is not None:
        # Show FLAIR background
        ax.imshow(
            np.rot90(flair_img[:, :, slice_idx]),
            cmap="gray"
        )
    else:
        # Show WMH only
        ax.imshow(
            np.rot90(img[:, :, slice_idx]),
            cmap="gray",
            interpolation="none"
        )

    # Overlay WMH mask
    if show_wmh:
        mask = np.rot90(img[:, :, slice_idx])

        # Only overlay if there are WMH voxels in this slice
        if np.any(mask):
            ax.imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap="autumn",
                alpha=0.5,
                interpolation="none"
            )

    ax.axis("off")
    st.pyplot(fig)

