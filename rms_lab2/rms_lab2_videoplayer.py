import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QSlider, QLabel,
    QHBoxLayout, QVBoxLayout, QFileDialog, QLineEdit
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


class MediaPlayer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Інф. системи – Медіаплеєр на Python (ручна реалізація)")
        self.resize(900, 500)

        # --- Медіаплеєр ---
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput()   # без родителя, чтобы избежать багов
        self.player.setAudioOutput(self.audio_output)

        # Відео-виджет (для відеофайлів)
        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        # --- Кнопки ---
        self.btn_open = QPushButton("Відкрити файл")
        self.btn_play = QPushButton("Play")
        self.btn_back = QPushButton("⏪ -5 c")
        self.btn_forward = QPushButton("⏩ +5 c")
        self.btn_stop = QPushButton("Restart")

        # Повзунок позиції
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)

        # Мітка часу
        self.time_label = QLabel("00:00 / 00:00")

        # Поле для URL (потокове медіа)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Встав URL потокового медіа і натисни Enter")

        # --- Layout-и ---
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.btn_open)
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_back)
        controls_layout.addWidget(self.btn_forward)
        controls_layout.addWidget(self.btn_stop)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_widget)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.position_slider)
        main_layout.addWidget(self.time_label)
        main_layout.addWidget(self.url_input)

        self.setLayout(main_layout)

        # --- Зв'язування сигналів ---
        self.btn_open.clicked.connect(self.open_file)
        self.btn_play.clicked.connect(self.play_pause)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_back.clicked.connect(lambda: self.seek_relative(-5000))
        self.btn_forward.clicked.connect(lambda: self.seek_relative(5000))

        self.position_slider.sliderMoved.connect(self.set_position)

        self.url_input.returnPressed.connect(self.play_stream)

        # Оновлення позиції/тривалості
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)

    # ====== Обробники подій ======

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть медіафайл",
            "",
            "Media files (*.mp3 *.wav *.mp4 *.avi *.mkv);;All files (*.*)"
        )
        if file_name:
            self.player.setSource(QUrl.fromLocalFile(file_name))
            self.player.play()
            self.btn_play.setText("Pause")

    def play_stream(self):
        url_text = self.url_input.text().strip()
        if url_text:
            url = QUrl(url_text)
            if url.isValid():
                self.player.setSource(url)
                self.player.play()
                self.btn_play.setText("Pause")

    def play_pause(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setText("Play")
        else:
            self.player.play()
            self.btn_play.setText("Pause")

    def stop(self):
        # Останавливаем и явно возвращаемся в начало
        self.player.stop()
        self.player.setPosition(0)
        self.btn_play.setText("Play")
        self.position_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")

    def seek_relative(self, offset_ms: int):
        """Промотка на offset_ms вперёд/назад (в мс)."""
        if self.player.duration() <= 0:
            return
        new_pos = self.player.position() + offset_ms
        if new_pos < 0:
            new_pos = 0
        if new_pos > self.player.duration():
            new_pos = self.player.duration()
        self.player.setPosition(new_pos)

    def set_position(self, position):
        # позиція в мілісекундах
        self.player.setPosition(position)

    def on_position_changed(self, position):
        self.position_slider.setValue(position)
        self.update_time_label()

    def on_duration_changed(self, duration):
        self.position_slider.setRange(0, duration)
        self.update_time_label()

    def update_time_label(self):
        pos = self.player.position()
        dur = self.player.duration()

        def ms_to_str(ms: int) -> str:
            if ms <= 0:
                return "00:00"
            s = ms // 1000
            m = s // 60
            s = s % 60
            return f"{m:02d}:{s:02d}"

        self.time_label.setText(f"{ms_to_str(pos)} / {ms_to_str(dur)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MediaPlayer()
    w.show()
    sys.exit(app.exec())
