import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import dill as pkl
import base64


@st.cache
def weights_to_distribution(all_weights_list):
    nepochs = len(all_weights_list)
    nlayers = int(len(all_weights_list[0])/2)
    ngpts = 500
    distributions = np.zeros((nepochs-1, nlayers, ngpts), dtype=np.float64)
    entropies = np.zeros((nepochs-1, nlayers), dtype=np.float64)
    xvals = np.zeros((nepochs-1, nlayers, ngpts), dtype=np.float64)
    diffs = np.zeros((nepochs-1, nlayers), dtype=np.float64)
    for ll in range(nepochs-1):
        for jj in range(nlayers):
            #weights = all_weights_list[ll][2*jj].numpy()-all_weights_list[-1][2*jj].numpy()
            #bias = all_weights_list[ll][2*jj+1].numpy()-all_weights_list[-1][2*jj+1].numpy()
            weights = all_weights_list[ll][2*jj].numpy()
            bias = all_weights_list[ll][2*jj+1].numpy()
            allterms = np.concatenate((weights.flatten(), bias))
            diffs[ll,jj] = np.linalg.norm(allterms)/np.linalg.norm(all_weights_list[-1][2*jj].numpy())
            xval, dist = FFTKDE(kernel='epa', bw='ISJ').fit(allterms).evaluate(ngpts)
            distributions[ll, jj, :] = dist
            xvals[ll, jj, :] = xval
            entropies[ll, jj] = -np.mean(np.ma.log(dist))
    return nepochs, nlayers, xvals, distributions, entropies, diffs

st.set_page_config(page_title="Entropy Application", 
                   page_icon="shinto_shrine",
                   layout="centered",
                   initial_sidebar_state="collapsed", 
                   menu_items={
                    "Get Help":None,
                    "Report a Bug": None,
                    "About":None})

st.markdown("<h1 style='text-align: center;'>Entropy Analysis</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a '.pkl' file that is a list of lists of tensorflow tensors.")

if uploaded_file is not None and uploaded_file.name.endswith(".pkl"):
    weights_list = pkl.load(uploaded_file)["all_weights_list"]
    st.write(f"File read: {uploaded_file.name}")

    nepochs, nlayers, supports, dists, entropies, diffs = weights_to_distribution(weights_list)

    epochs = np.arange(nepochs)[1:]

    fig, ax = plt.subplots(1,2,figsize=(16,9))

    ax[0].plot(epochs, entropies, '.')
    ax[0].set(yscale="log")
    ax[1].plot(epochs, diffs, '.')
    ax[1].set(yscale="log")

    st.pyplot(fig)

else:
    st.write(f"No file read.")
    weights_list = None



