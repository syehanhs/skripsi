import cv2
import numpy as np
import av
import time
import streamlit as st
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from google import genai
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- 1. MENGATUR UI DENGAN STREAMLIT ---
st.set_page_config(
    page_title="Hand Gesture with Gemini AI for Physics",
    page_icon="🤖",
    layout="wide"
)

# CSS murni hanya untuk merapikan kotak jawaban (tanpa membesarkan kamera)
st.markdown("""
    <style>
    .answer-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #ff4b4b;
        color: #000000;
        min-height: 200px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Vision Solver dengan Hand Gesture")
st.markdown("Arahkan tangan ke kamera untuk menggambar persoalan, lalu biarkan AI menyelesaikannya!")
st.divider()

# --- 2. SIDEBAR & PANDUAN ---
with st.sidebar:
    st.header("⚙️ Kontrol Panel")
    st.markdown("Klik tombol **START** di bawah kamera untuk memulai.")
    st.markdown("---")
    st.header("🖐️ Panduan Gestur Jari")
    st.info("**1 Jari (Telunjuk):**\nMenggambar di layar.")
    st.warning("**3 Jari (Tengah, Manis, Kelingking):**\nMenghapus seluruh gambar (Clear Canvas).")
    st.success("**Gestur 'Rock' (Jempol & Kelingking):**\nTahan 2 detik untuk mengirim gambar ke AI.")

# --- 3. KONFIGURASI API & WEBRTC ---
API_KEY_AMAN = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY_AMAN)
model = "gemini-2.5-flash-lite"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# --- 4. CLASS PEMROSESAN VIDEO (WEBRTC) ---
class HandDetectorProcessor(VideoProcessorBase):
    def __init__(self):
        # Inisialisasi detector di dalam class agar aman dari refresh UI Streamlit
        self.detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
        self.prev_pos = None
        self.canvas = None
        self.gesture_start_time = 0
        self.gesture_active = False
        self.has_sent = False
        self.HOLD_DURATION = 2.0
        self.ai_response = "Jawaban akan muncul di sini..."

    def getHandInfo(self, img):
        # Parameter flipType=False karena image sudah di-flip di fungsi recv
        hands, img = self.detector.findHands(img, draw=True, flipType=True)
        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = self.detector.fingersUp(hand)
            return fingers, lmList
        else:
            return None

    def sendToAI(self, canvas, fingers):
        if fingers == [1, 0, 0, 0, 1]:
            pil_image = Image.fromarray(canvas)
            try:
                prompt_text = (
                    "Kamu adalah asisten pengajar Fisika dasar. "
                    "Gambar ini adalah coretan tangan tentang persoalan fisika. "
                    "Berikan jawaban akhir dan langkah pengerjaan dengan singkat. "
                    "Jangan berikan salam pembuka atau penutup."
                )
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt_text, pil_image]
                )
                return response.text
            except Exception as e:
                # TAMPILKAN ERROR KE TERMINAL JUGA
                print("ERROR GEMINI:", str(e))
                return f"[Error Gemini] {str(e)}"
        return ""

    # def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
    #     img = frame.to_ndarray(format="bgr24")
    #     img = cv2.flip(img, 1)
    #
    #     if self.canvas is None or self.canvas.shape != img.shape:
    #         self.canvas = np.zeros_like(img)
    #
    #     # Memanggil logika modular Anda
    #     info = self.getHandInfo(img)
    #
    #     if info:
    #         fingers, lmList = info
    #
    #         # --- LOGIKA MENGGAMBAR (Membaca koordinat Y agar mulus dan tidak putus) ---
    #         # lmList[8][1] = Kordinat Y ujung telunjuk | lmList[6][1] = Kordinat Y pangkal telunjuk
    #         index_is_up = lmList[8][1] < lmList[6][1]
    #         middle_is_down = lmList[12][1] > lmList[10][1]
    #
    #         if index_is_up and middle_is_down:
    #             current_pos = lmList[8][0:2]
    #             if self.prev_pos is None:
    #                 self.prev_pos = current_pos
    #
    #             # Menggambar garis tebal dan halus (LINE_AA)
    #             cv2.line(self.canvas, current_pos, self.prev_pos, (255, 0, 255), 10, cv2.LINE_AA)
    #             self.prev_pos = current_pos
    #
    #         # --- LOGIKA MENGHAPUS ---
    #         elif fingers == [0, 1, 1, 1, 0]:
    #             self.canvas = np.zeros_like(img)
    #             self.prev_pos = None
    #             self.ai_response = "Canvas dibersihkan. Silakan menggambar lagi."
    #
    #         else:
    #             self.prev_pos = None
    #
    #         # --- LOGIKA MENGIRIM KE AI ---
    #         if fingers == [1, 0, 0, 0, 1]:
    #             if not self.gesture_active:
    #                 self.gesture_start_time = time.time()
    #                 self.gesture_active = True
    #                 self.has_sent = False
    #
    #             time_held = time.time() - self.gesture_start_time
    #             cv2.putText(img, f"Tahan... {int(time_held)}s/2s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
    #                         2)
    #
    #             if time_held > self.HOLD_DURATION and not self.has_sent:
    #                 cv2.putText(img, "MENGIRIM KE AI...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #                 self.ai_response = self.sendToAI(self.canvas, fingers)
    #                 self.has_sent = True
    #         else:
    #             self.gesture_active = False
    #             self.gesture_start_time = 0
    #             self.has_sent = False
    #
    #     # Menggabungkan canvas gambar dengan frame kamera (opacity tinta 0.9)
    #     image_combined = cv2.addWeighted(img, 1.0, self.canvas, 0.9, 0)
    #     return av.VideoFrame.from_ndarray(image_combined, format="bgr24")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.canvas is None or self.canvas.shape != img.shape:
            self.canvas = np.zeros_like(img)

        info = self.getHandInfo(img)

        if info:
            fingers, lmList = info

            # --- MENAMPILKAN MATRIX GESTURE DI LAYAR ---
            # Teks warna biru (255, 0, 0) di pojok kiri atas
            cv2.putText(img, f"Matrix: {fingers}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # --- LOGIKA MENGGAMBAR ---
            index_is_up = lmList[8][1] < lmList[6][1]
            middle_is_down = lmList[12][1] > lmList[10][1]

            if index_is_up and middle_is_down:
                current_pos = lmList[8][0:2]
                if self.prev_pos is None:
                    self.prev_pos = current_pos
                cv2.line(self.canvas, current_pos, self.prev_pos, (255, 0, 255), 10, cv2.LINE_AA)
                self.prev_pos = current_pos

            # --- LOGIKA MENGHAPUS ---
            elif fingers == [0, 1, 1, 1, 0]:
                self.canvas = np.zeros_like(img)
                self.prev_pos = None
                self.ai_response = "Canvas dibersihkan. Silakan menggambar lagi."

            else:
                self.prev_pos = None

            # --- LOGIKA MENGIRIM KE AI ---
            if fingers == [1, 0, 0, 0, 1]:
                if not self.gesture_active:
                    self.gesture_start_time = time.time()
                    self.gesture_active = True
                    self.has_sent = False

                time_held = time.time() - self.gesture_start_time
                # Posisi Y diturunkan ke 80 agar tidak menabrak teks Matrix
                cv2.putText(img, f"Tahan... {int(time_held)}s/2s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2)

                if time_held > self.HOLD_DURATION and not self.has_sent:
                    # Posisi Y diturunkan ke 120
                    cv2.putText(img, "MENGIRIM KE AI...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.ai_response = self.sendToAI(self.canvas, fingers)
                    self.has_sent = True
            else:
                self.gesture_active = False
                self.gesture_start_time = 0
                self.has_sent = False

        image_combined = cv2.addWeighted(img, 1.0, self.canvas, 0.9, 0)
        return av.VideoFrame.from_ndarray(image_combined, format="bgr24")

# --- 5. TATA LETAK UI (KOLOM KIRI & KANAN SEPERTI AWAL) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Tampilan Kamera")
    ctx = webrtc_streamer(
        key="hand-gesture-ai",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HandDetectorProcessor,
        # Ukuran standar default
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("💡 Jawaban AI")
    output_text_area = st.empty()

    if ctx.video_processor:
        answer_to_show = ctx.video_processor.ai_response
        output_text_area.markdown(f'<div class="answer-box">{answer_to_show}</div>', unsafe_allow_html=True)
    else:
        output_text_area.markdown('<div class="answer-box">Nyalakan kamera (Klik START) untuk memulai...</div>',
                                  unsafe_allow_html=True)

    st.caption("Jika Anda sudah mengirim gambar namun jawaban belum berubah, klik tombol di bawah ini.")
    if st.button("🔄 Segarkan Jawaban"):
        pass