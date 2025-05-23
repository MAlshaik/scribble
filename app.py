import streamlit as st

from streamlit_drawable_canvas import st_canvas

# Specify canvas parameters in application

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)

stroke_color = st.sidebar.color_picker("Stroke color hex: ")

bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component

canvas_result = st_canvas(

    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity

    stroke_width=stroke_width,

    stroke_color=stroke_color,

    background_color=bg_color,

    update_streamlit=realtime_update,

    height=500,

    drawing_mode="freedraw",

    point_display_radius=0,

    key="canvas",

)

# Do something interesting with the image data and paths

# The image from the canvas is available as an ndarray and can be accessed via "canvas_result.image_data"

#if canvas_result.image_data is not None:


