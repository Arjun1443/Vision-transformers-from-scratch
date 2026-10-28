import io,hashlib,threading
import numpy as np
import streamlit as st
from PIL import Image,ImageOps
from ui import page

page('Vision Transformer Lens','Inspect how a pretrained Vision Transformer distributes final-layer attention over image patches.','EXPLAINABLE AI')
@st.cache_resource
def load_model():
    import timm
    from timm.data import resolve_model_data_config,create_transform
    model=timm.create_model('vit_base_patch16_224',pretrained=True)
    model.eval()
    model.blocks[-1].attn.fused_attn=False
    transform=create_transform(**resolve_model_data_config(model),is_training=False)
    return model,transform,threading.Lock()
upload=st.file_uploader('Upload an image to inspect',type=['png','jpg','jpeg'])
opacity=st.slider('Attention overlay strength',0.0,1.0,.5,.05)
if upload:
    image_id=hashlib.sha256(upload.getvalue()).hexdigest()
    if st.session_state.get('image_id')!=image_id:
        st.session_state.pop('overlay',None)
        st.session_state.image_id=image_id
    try: image=ImageOps.exif_transpose(Image.open(upload)).convert('RGB')
    except Exception: st.error('Please upload a valid image.'); st.stop()
    st.image(image,width=360)
    if st.button('Generate attention overlay',type='primary'):
        try:
            import torch
            with st.spinner('Loading ViT and extracting attention…'):
                model,transform,lock=load_model()
                with lock:
                    captured=[]
                    hook=model.blocks[-1].attn.attn_drop.register_forward_hook(lambda module,inputs,output:captured.append(output.detach().cpu()))
                    try:
                        with torch.inference_mode(): model(transform(image).unsqueeze(0))
                    finally: hook.remove()
                attention=captured[0][0].mean(0)[0,1:].numpy().reshape(14,14)
                attention=(attention-attention.min())/(np.ptp(attention)+1e-8)
                base=transform(image)
                # Show the same center crop passed to the model, avoiding a misleading alignment.
                config=__import__('timm').data.resolve_model_data_config(model)
                mean=np.asarray(config['mean'])[:,None,None]
                std=np.asarray(config['std'])[:,None,None]
                pixels=np.clip((base.numpy()*std+mean)*255,0,255).astype('uint8').transpose(1,2,0)
                crop=Image.fromarray(pixels)
                heat=Image.fromarray(np.uint8(attention*255)).resize(crop.size,Image.Resampling.BILINEAR)
                strength=np.asarray(heat)/255*opacity
                rgb=np.asarray(crop,dtype=float)
                tint=np.zeros_like(rgb); tint[...,0]=255; tint[...,1]=80
                overlay=np.uint8(rgb*(1-strength[...,None])+tint*strength[...,None])
                output=io.BytesIO(); Image.fromarray(overlay).save(output,format='PNG')
                st.session_state.overlay=output.getvalue()
        except Exception:
            st.error('Could not extract attention. Check model-download access, timm compatibility, and available memory.')
    if 'overlay' in st.session_state:
        st.image(st.session_state.overlay,caption='Final-layer CLS attention on the model’s 224×224 center crop',width=448)
        st.download_button('Download overlay',st.session_state.overlay,'attention-overlay.png','image/png')
with st.expander('How to interpret this visualization',expanded=True):
    st.write('The app averages attention heads from the final block and displays CLS-to-patch weights. This is a pretrained ImageNet ViT, not a medical diagnostic model. Attention does not establish causal explanation or pathology localization. The notebook contains the annotation/IoU experiments; this interface does not claim IoU results without ground truth.')
    st.caption('The first run downloads pretrained weights. This model may exceed the memory budget of a small Community Cloud instance.')
