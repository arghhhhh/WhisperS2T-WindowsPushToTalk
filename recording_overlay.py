"""
Recording overlay widget - a small pill-shaped overlay with animated audio waveform.

Shows on the bottom-right of the screen above the taskbar while recording is active.
Runs Qt in a dedicated background thread so it doesn't block the main hotkey loop.
"""

import sys
import struct
import threading
import numpy as np
from collections import deque

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QLinearGradient, QPen, QBrush, QFont
)


# =============================================================================
# CONSTANTS
# =============================================================================

PILL_WIDTH = 200
PILL_HEIGHT = 48
PILL_RADIUS = 24
MARGIN_RIGHT = 20
MARGIN_BOTTOM = 20

NUM_BARS = 28
BAR_WIDTH = 3
BAR_GAP = 2
BAR_MIN_HEIGHT = 3
BAR_MAX_HEIGHT = 28

FPS = 30
DECAY = 0.85  # How quickly bars fall back down

# Colors
BG_COLOR = QColor(24, 24, 28, 230)
BAR_COLOR_LOW = QColor(80, 200, 120)    # Green
BAR_COLOR_HIGH = QColor(255, 90, 90)    # Red for loud
LABEL_COLOR = QColor(255, 255, 255, 200)
BORDER_COLOR = QColor(255, 255, 255, 30)

# Recording dot animation
DOT_COLOR = QColor(255, 60, 60)


# =============================================================================
# BRIDGE for cross-thread communication
# =============================================================================

class _OverlayBridge(QObject):
    """Qt signal bridge for thread-safe communication."""
    show_signal = Signal()
    hide_signal = Signal()
    audio_signal = Signal(bytes)


# =============================================================================
# OVERLAY WIDGET
# =============================================================================

class _PillOverlay(QWidget):
    """The actual pill-shaped overlay widget."""

    def __init__(self):
        super().__init__()

        # Frameless, transparent, always on top, tool window (no taskbar entry)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint
            | Qt.FramelessWindowHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(PILL_WIDTH, PILL_HEIGHT)

        # Bar heights (smoothed)
        self.bar_heights = [BAR_MIN_HEIGHT] * NUM_BARS
        self.target_heights = [BAR_MIN_HEIGHT] * NUM_BARS

        # Recording dot blink
        self.dot_visible = True
        self.dot_timer = 0

        # Animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._tick)
        self.anim_timer.setInterval(1000 // FPS)

        # Position above taskbar, bottom-right
        self._reposition()

    def _reposition(self):
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            x = avail.right() - PILL_WIDTH - MARGIN_RIGHT
            y = avail.bottom() - PILL_HEIGHT - MARGIN_BOTTOM
            self.move(x, y)

    def showEvent(self, event):
        super().showEvent(event)
        self.anim_timer.start()

    def hideEvent(self, event):
        self.anim_timer.stop()
        self.bar_heights = [BAR_MIN_HEIGHT] * NUM_BARS
        self.target_heights = [BAR_MIN_HEIGHT] * NUM_BARS
        super().hideEvent(event)

    def feed_audio(self, raw_bytes: bytes):
        """Feed raw int16 audio bytes to update waveform targets."""
        count = len(raw_bytes) // 2
        if count == 0:
            return
        shorts = struct.unpack(f'{count}h', raw_bytes)
        samples = np.abs(np.array(shorts, dtype=np.float32)) / 32768.0

        # Split samples into NUM_BARS buckets and take RMS of each
        bucket_size = max(1, len(samples) // NUM_BARS)
        for i in range(NUM_BARS):
            start = i * bucket_size
            end = min(start + bucket_size, len(samples))
            if start >= len(samples):
                break
            rms = float(np.sqrt(np.mean(samples[start:end] ** 2)))
            # Log scale so quiet speech is still visible (rms 0.01-0.15 typical)
            if rms > 0.001:
                normalized = (np.log10(rms) + 3) / 2.5  # maps ~0.001→0.0, ~0.18→1.0
                normalized = max(0.0, min(1.0, normalized))
            else:
                normalized = 0.0
            h = BAR_MIN_HEIGHT + normalized * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT)
            self.target_heights[i] = max(self.target_heights[i], h)

    def _tick(self):
        """Animation frame - smooth bars toward targets, then decay."""
        self.dot_timer += 1
        if self.dot_timer >= FPS // 2:
            self.dot_visible = not self.dot_visible
            self.dot_timer = 0

        for i in range(NUM_BARS):
            # Lerp toward target
            self.bar_heights[i] += (self.target_heights[i] - self.bar_heights[i]) * 0.4
            # Decay target
            self.target_heights[i] *= DECAY
            if self.target_heights[i] < BAR_MIN_HEIGHT:
                self.target_heights[i] = BAR_MIN_HEIGHT

        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # -- Pill background --
        pill = QPainterPath()
        pill.addRoundedRect(QRectF(0, 0, PILL_WIDTH, PILL_HEIGHT), PILL_RADIUS, PILL_RADIUS)

        p.fillPath(pill, QBrush(BG_COLOR))
        p.setPen(QPen(BORDER_COLOR, 1))
        p.drawPath(pill)

        # -- Recording dot --
        dot_x = 16
        dot_y = PILL_HEIGHT / 2
        if self.dot_visible:
            p.setBrush(QBrush(DOT_COLOR))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(dot_x, dot_y), 5, 5)
        # Dim dot when blinking off
        else:
            p.setBrush(QBrush(QColor(255, 60, 60, 80)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(dot_x, dot_y), 5, 5)

        # -- REC label --
        p.setPen(QPen(LABEL_COLOR))
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(28, int(PILL_HEIGHT / 2 + 4), "REC")

        # -- Waveform bars --
        bars_start_x = 60
        bars_area_width = PILL_WIDTH - bars_start_x - 12
        actual_bar_total = BAR_WIDTH + BAR_GAP
        center_y = PILL_HEIGHT / 2

        for i in range(NUM_BARS):
            x = bars_start_x + i * actual_bar_total
            if x + BAR_WIDTH > PILL_WIDTH - 8:
                break

            h = self.bar_heights[i]

            # Color gradient based on height
            t = (h - BAR_MIN_HEIGHT) / (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT) if BAR_MAX_HEIGHT > BAR_MIN_HEIGHT else 0
            r = int(BAR_COLOR_LOW.red() + t * (BAR_COLOR_HIGH.red() - BAR_COLOR_LOW.red()))
            g = int(BAR_COLOR_LOW.green() + t * (BAR_COLOR_HIGH.green() - BAR_COLOR_LOW.green()))
            b = int(BAR_COLOR_LOW.blue() + t * (BAR_COLOR_HIGH.blue() - BAR_COLOR_LOW.blue()))
            color = QColor(r, g, b, 220)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            rect = QRectF(x, center_y - h / 2, BAR_WIDTH, h)
            p.drawRoundedRect(rect, BAR_WIDTH / 2, BAR_WIDTH / 2)

        p.end()


# =============================================================================
# PUBLIC API - Thread-safe overlay manager
# =============================================================================

class RecordingOverlay:
    """
    Thread-safe recording overlay manager.

    Usage:
        overlay = RecordingOverlay()
        overlay.show()                    # Show the pill overlay
        overlay.feed_audio(raw_bytes)     # Feed raw int16 PCM bytes for waveform
        overlay.hide()                    # Hide the overlay
        overlay.shutdown()                # Clean up Qt (call on app exit)
    """

    def __init__(self):
        self._app = None
        self._widget = None
        self._bridge = None
        self._thread = None
        self._ready = threading.Event()

        # Start Qt in background thread
        self._thread = threading.Thread(target=self._run_qt, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_qt(self):
        """Run Qt event loop in background thread."""
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)

        self._widget = _PillOverlay()
        self._bridge = _OverlayBridge()

        # Connect signals
        self._bridge.show_signal.connect(self._widget.show)
        self._bridge.hide_signal.connect(self._widget.hide)
        self._bridge.audio_signal.connect(self._widget.feed_audio)

        self._ready.set()
        self._app.exec()

    def show(self):
        """Show the overlay (thread-safe)."""
        if self._bridge:
            self._bridge.show_signal.emit()

    def hide(self):
        """Hide the overlay (thread-safe)."""
        if self._bridge:
            self._bridge.hide_signal.emit()

    def feed_audio(self, raw_bytes: bytes):
        """Feed raw int16 PCM audio bytes for waveform visualization (thread-safe)."""
        if self._bridge:
            self._bridge.audio_signal.emit(raw_bytes)

    def shutdown(self):
        """Shut down the Qt event loop."""
        if self._app:
            self._app.quit()
