from itertools import count
import cv2
from PIL.features import check
from numpy.f2py.crackfortran import verbose
from torch.distributed.tensor import empty
from ultralytics import YOLO
import streamlit as st
import time
import numpy as np
import tempfile

st.title('YOLO8 + Streamlit + OpenCV')

st.sidebar.header('Меню')
model_name = st.sidebar.selectbox('Выберите модель Yolo', ['yolov8n.pt', 'yolov11n.pt'])
conf_value = st.sidebar.slider('Точность:', 0.10, 0.95, 0.5, 0.05)
type_button = st.sidebar.radio('Режим работы:', ['Photo', 'Video', 'Camera'])
start_button = st.sidebar.button('Запустить')

model = YOLO(model_name)
all_classes = list(model.names.values())

select_classes = st.sidebar.multiselect('Выберите классы для детекции',
                                        all_classes, default=['person', 'airplane'])
st.sidebar.info(f'Выбрано классов: {len(select_classes)}')

def check_frame(frame):
    result = model(frame, stream=True, conf=conf_value, verbose=False)
    counts = {}

    for i in result:
        for n in i.boxes:
            cls = int(n.cls[0])
            label = model.names[cls]
            conf = round(float(n.conf[0]), 2)
            counts[label] = counts.get(label, 0)+1

            if label not in select_classes:
                continue

            x, y, w, h = map(int, n.xyxy[0])
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}, {conf*100}%', (x, y -10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame, counts

if type_button == 'Photo':
    image_file = st.file_uploader('Загрузите изображение', type={'jpg', 'jpeg', 'png'})
    if image_file and start_button:
        image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        frame_class, counts = check_frame(frame)
        frame_rgb = cv2.cvtColor(frame_class, cv2.COLOR_BGRA2RGB)
        st.image(frame_rgb, caption='Result Yolo model', use_container_width=True)
        st.write(f'Обнаружено объектов: {sum(counts.values())}')
        st.json(counts)


elif type_button == 'Video':
    video_file = st.file_uploader('Загрузите video', type={'mp4'})
    if video_file and start_button:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        cap = cv2.VideoCapture(temp_file.name)
        check_video = st.empty()

        total_counts = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_class, counts = check_frame(frame)
            frame_rgb = cv2.cvtColor(frame_class, cv2.COLOR_BGRA2RGB)
            check_video.image(frame_rgb)

            # for k, v in counts.items():
            #     total_counts[k] = total_counts.get(k, 0) + v

        cap.release()
        st.write(f'Всего объектов найдено: {sum(total_counts.values())}')
        st.json(total_counts)


elif type_button == 'Camera' and start_button:
    if st.sidebar.button("Остановить камеру"):
        st.session_state.stop_camera = True

    cap = cv2.VideoCapture(0)
    start_fps = 0
    check_video = st.empty()
    info_video = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning('Camera not found')
            break

        frame_class, counts = check_frame(frame)
        end_fps = time.time()
        fps = 1 / (end_fps - start_fps) if start_fps != 0 else 0
        start_fps = end_fps

        cv2.putText(frame, f'FPS: {round(fps, 1)}',
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 128, 0), 2)

        cv2.putText(frame, f'Objects: {sum(counts.values())}',
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 128, 255), 2)

        frame_rgb = cv2.cvtColor(frame_class, cv2.COLOR_BGRA2RGB)
        check_video.image(frame_rgb)


        info_video.markdown(
            f"""FPS: {fps:.2f},
                Объекты: {sum(counts.values())},
                Kлассы: {':'.join([f'{k}, {v}' for k, v in counts.items()])}"""
        )


    cap.release()

else:
    st.info('Выберите режим классы и нажмите запустить')




