import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import csv
from datetime import datetime
import os
import threading

# For audio alerts
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# For graph generation
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Install matplotlib for reports: pip install matplotlib")

class ReportGenerator:
    def __init__(self, log_filename, session_start, alert_system=None):
        self.log_filename = log_filename
        self.session_start = session_start
        self.alert_system = alert_system
        
    def generate_report(self):
        """Generate comprehensive session report with graphs"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Cannot generate graphs - matplotlib not installed")
            return
        
        print("\n🎨 Generating visual report...")
        
        # Read data from CSV
        data = self.read_log_data()
        if not data:
            print("❌ No data to generate report")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('FocusSense Session Report', fontsize=20, fontweight='bold')
        
        # 1. Attention over time (line chart)
        ax1 = plt.subplot(2, 3, 1)
        self.plot_attention_timeline(ax1, data)
        
        # 2. Status distribution (pie chart)
        ax2 = plt.subplot(2, 3, 2)
        self.plot_status_distribution(ax2, data)
        
        # 3. Metrics over time (multi-line)
        ax3 = plt.subplot(2, 3, 3)
        self.plot_metrics_timeline(ax3, data)
        
        # 4. Attention histogram
        ax4 = plt.subplot(2, 3, 4)
        self.plot_attention_histogram(ax4, data)
        
        # 5. Statistics summary (text)
        ax5 = plt.subplot(2, 3, 5)
        self.plot_statistics_summary(ax5, data)
        
        # 6. Hourly performance
        ax6 = plt.subplot(2, 3, 6)
        self.plot_hourly_performance(ax6, data)
        
        plt.tight_layout()
        
        # Save report
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        report_filename = f"logs/report_{timestamp}.png"
        plt.savefig(report_filename, dpi=150, bbox_inches='tight')
        print(f"✅ Report saved: {report_filename}")
        
        # Generate HTML report
        html_filename = self.generate_html_report(data, report_filename)
        print(f"✅ HTML report: {html_filename}")
        
        # Open in browser
        import webbrowser
        webbrowser.open(html_filename)
        
        plt.close()
    
    def read_log_data(self):
        """Read and parse CSV log file"""
        data = {
            'timestamps': [],
            'elapsed': [],
            'attention': [],
            'status': [],
            'ear': [],
            'gaze': [],
            'head_angle': [],
            'blinks': [],
            'alerts': []
        }
        
        try:
            with open(self.log_filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data['timestamps'].append(datetime.strptime(row['Timestamp'], "%Y-%m-%d %H:%M:%S"))
                    data['elapsed'].append(int(row['Elapsed_Seconds']))
                    data['attention'].append(int(row['Attention_%']))
                    data['status'].append(row['Status'])
                    data['ear'].append(float(row['EAR']))
                    data['gaze'].append(float(row['Gaze_Deviation']))
                    data['head_angle'].append(float(row['Head_Angle']))
                    data['blinks'].append(int(row['Blink_Count']))
                    data['alerts'].append(row.get('Alert_Triggered', 'NO'))
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return None
        
        return data
    
    def plot_attention_timeline(self, ax, data):
        """Plot attention percentage over time"""
        ax.plot(data['elapsed'], data['attention'], linewidth=2, color='#2E86AB')
        ax.fill_between(data['elapsed'], data['attention'], alpha=0.3, color='#2E86AB')
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Highly Focused')
        ax.axhline(y=65, color='orange', linestyle='--', alpha=0.5, label='Focused')
        ax.axhline(y=45, color='red', linestyle='--', alpha=0.5, label='Distracted')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Attention %')
        ax.set_title('Attention Over Time', fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    def plot_status_distribution(self, ax, data):
        """Plot pie chart of status distribution"""
        status_counts = {}
        for status in data['status']:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        colors = {
            'HIGHLY FOCUSED': '#00b894',
            'FOCUSED': '#55efc4',
            'MODERATE': '#fdcb6e',
            'DISTRACTED': '#d63031'
        }
        
        labels = list(status_counts.keys())
        sizes = list(status_counts.values())
        chart_colors = [colors.get(label, '#95a5a6') for label in labels]
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=chart_colors, startangle=90)
        ax.set_title('Status Distribution', fontweight='bold')
    
    def plot_metrics_timeline(self, ax, data):
        """Plot EAR, Gaze, and Head Angle over time"""
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        p1, = ax.plot(data['elapsed'], data['ear'], 'g-', linewidth=1.5, label='EAR', alpha=0.7)
        p2, = ax2.plot(data['elapsed'], data['gaze'], 'b-', linewidth=1.5, label='Gaze', alpha=0.7)
        p3, = ax3.plot(data['elapsed'], data['head_angle'], 'r-', linewidth=1.5, label='Head Angle', alpha=0.7)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('EAR', color='g')
        ax2.set_ylabel('Gaze Deviation', color='b')
        ax3.set_ylabel('Head Angle (°)', color='r')
        
        ax.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='b')
        ax3.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('Metrics Timeline', fontweight='bold')
        ax.legend(handles=[p1, p2, p3], loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_attention_histogram(self, ax, data):
        """Plot histogram of attention distribution"""
        ax.hist(data['attention'], bins=20, color='#6c5ce7', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(data['attention']), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(data['attention']):.1f}%")
        ax.set_xlabel('Attention %')
        ax.set_ylabel('Frequency')
        ax.set_title('Attention Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_statistics_summary(self, ax, data):
        """Display statistics as text"""
        ax.axis('off')
        
        # Calculate statistics
        avg_attention = np.mean(data['attention'])
        max_attention = np.max(data['attention'])
        min_attention = np.min(data['attention'])
        std_attention = np.std(data['attention'])
        total_blinks = data['blinks'][-1] if data['blinks'] else 0
        total_alerts = data['alerts'].count('YES')
        
        # Calculate time in each state
        total_time = len(data['attention'])
        focused_time = sum(1 for a in data['attention'] if a >= 65)
        moderate_time = sum(1 for a in data['attention'] if 45 <= a < 65)
        distracted_time = sum(1 for a in data['attention'] if a < 45)
        
        stats_text = f"""
        📊 SESSION STATISTICS
        
        ⏱️  Duration: {total_time} seconds ({total_time//60}m {total_time%60}s)
        
        📈 Attention Metrics:
           • Average: {avg_attention:.1f}%
           • Maximum: {max_attention}%
           • Minimum: {min_attention}%
           • Std Dev: {std_attention:.1f}%
        
        ⏰ Time Distribution:
           • Focused: {focused_time}s ({focused_time/total_time*100:.1f}%)
           • Moderate: {moderate_time}s ({moderate_time/total_time*100:.1f}%)
           • Distracted: {distracted_time}s ({distracted_time/total_time*100:.1f}%)
        
        👁️  Total Blinks: {total_blinks}
        🔔 Alerts Triggered: {total_alerts}
        
        💡 Focus Score: {self.calculate_focus_score(data)}/100
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def plot_hourly_performance(self, ax, data):
        """Plot average attention by minute intervals"""
        if len(data['elapsed']) < 60:
            ax.text(0.5, 0.5, 'Need >60s data\nfor hourly view', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        # Group by minute
        minute_data = {}
        for elapsed, attention in zip(data['elapsed'], data['attention']):
            minute = elapsed // 60
            if minute not in minute_data:
                minute_data[minute] = []
            minute_data[minute].append(attention)
        
        minutes = sorted(minute_data.keys())
        avg_attention = [np.mean(minute_data[m]) for m in minutes]
        
        ax.bar(minutes, avg_attention, color='#0984e3', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Minute')
        ax.set_ylabel('Average Attention %')
        ax.set_title('Performance by Minute', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
    
    def calculate_focus_score(self, data):
        """Calculate overall focus score (0-100)"""
        avg_attention = np.mean(data['attention'])
        consistency = 100 - np.std(data['attention'])
        total_alerts = data['alerts'].count('YES')
        alert_penalty = min(20, total_alerts * 5)
        
        score = (avg_attention * 0.6 + consistency * 0.4) - alert_penalty
        return max(0, min(100, int(score)))
    
    def generate_html_report(self, data, image_path):
        """Generate HTML report with embedded image"""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        html_filename = f"logs/report_{timestamp}.html"
        
        avg_attention = np.mean(data['attention'])
        total_time = len(data['attention'])
        total_blinks = data['blinks'][-1] if data['blinks'] else 0
        total_alerts = data['alerts'].count('YES')
        focus_score = self.calculate_focus_score(data)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FocusSense Report - {self.session_start.strftime("%Y-%m-%d %H:%M:%S")}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                h1 {{
                    color: #2d3436;
                    text-align: center;
                    margin-bottom: 10px;
                }}
                .date {{
                    text-align: center;
                    color: #636e72;
                    margin-bottom: 30px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                .stat-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .stat-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .chart-container {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .chart-container img {{
                    max-width: 100%;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: #636e72;
                    font-size: 0.9em;
                }}
                .score-badge {{
                    display: inline-block;
                    font-size: 3em;
                    font-weight: bold;
                    padding: 20px 40px;
                    border-radius: 50%;
                    background: {'#00b894' if focus_score >= 80 else '#fdcb6e' if focus_score >= 60 else '#d63031'};
                    color: white;
                    margin: 20px auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 FocusSense Session Report</h1>
                <div class="date">{self.session_start.strftime("%A, %B %d, %Y at %H:%M:%S")}</div>
                
                <div style="text-align: center;">
                    <div class="score-badge">{focus_score}</div>
                    <h2>Focus Score</h2>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">⏱️ Duration</div>
                        <div class="stat-value">{total_time//60}m {total_time%60}s</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">📈 Avg Attention</div>
                        <div class="stat-value">{avg_attention:.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">👁️ Blinks</div>
                        <div class="stat-value">{total_blinks}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🔔 Alerts</div>
                        <div class="stat-value">{total_alerts}</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>📊 Detailed Analytics</h2>
                    <img src="{os.path.basename(image_path)}" alt="Session Analytics">
                </div>
                
                <div class="footer">
                    <p>Generated by FocusSense AI Attention Detection System</p>
                    <p>Report generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_filename

class AlertSystem:
    def __init__(self):
        self.distraction_start_time = None
        self.distraction_threshold = 10
        self.alert_cooldown = 30
        self.last_alert_time = 0
        self.total_alerts = 0
        self.distraction_events = []
        self.is_alerting = False
        
    def check_attention(self, attention_score):
        current_time = time.time()
        
        if attention_score < 45:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            
            distraction_duration = current_time - self.distraction_start_time
            
            if (distraction_duration >= self.distraction_threshold and 
                current_time - self.last_alert_time > self.alert_cooldown):
                self.trigger_alert()
                self.last_alert_time = current_time
                self.total_alerts += 1
                self.distraction_events.append({
                    'timestamp': datetime.now(),
                    'duration': distraction_duration
                })
                return True
        else:
            self.distraction_start_time = None
            self.is_alerting = False
        
        return False
    
    def trigger_alert(self):
        self.is_alerting = True
        if AUDIO_AVAILABLE:
            threading.Thread(target=self.play_alert_sound, daemon=True).start()
    
    def play_alert_sound(self):
        try:
            if os.name == 'nt':
                winsound.Beep(1000, 500)
                time.sleep(0.3)
                winsound.Beep(1000, 500)
        except:
            pass

class FocusSense:
    def __init__(self, enable_logging=True, enable_alerts=True):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        self.attention_history = deque(maxlen=30)
        self.blink_counter = 0
        self.blink_threshold = 0.19
        self.last_blink_time = time.time()
        
        self.GAZE_THRESHOLD = 0.55
        self.HEAD_POSE_THRESHOLD = 25
        self.BLINK_RATE_MIN = 0.15
        self.BLINK_RATE_MAX = 0.6
        
        self.enable_logging = enable_logging
        self.log_interval = 1.0
        self.last_log_time = time.time()
        self.session_start = datetime.now()
        
        self.enable_alerts = enable_alerts
        self.alert_system = AlertSystem() if enable_alerts else None
        
        if self.enable_logging:
            self.setup_logging()
    
    def setup_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"logs/focussense_session_{timestamp}.csv"
        
        with open(self.log_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Elapsed_Seconds', 'Attention_%', 'Status',
                'EAR', 'Gaze_Deviation', 'Head_Angle', 'Blink_Count', 'Alert_Triggered'
            ])
        
        print(f"📝 Logging enabled: {self.log_filename}")
    
    def log_data(self, attention_score, status, ear, gaze, head_angle, alert_triggered=False):
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_interval:
            elapsed = int(current_time - time.mktime(self.session_start.timetuple()))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.log_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, elapsed, attention_score, status,
                    f"{ear:.3f}", f"{gaze:.3f}", f"{head_angle:.2f}", 
                    self.blink_counter, 'YES' if alert_triggered else 'NO'
                ])
            
            self.last_log_time = current_time
    
    def calculate_ear(self, eye_points):
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        return (v1 + v2) / (2.0 * h)
    
    def get_eye_points(self, landmarks, indices, img_w, img_h):
        points = []
        for idx in indices:
            point = landmarks[idx]
            points.append([point.x * img_w, point.y * img_h])
        return np.array(points)
    
    def get_iris_position(self, landmarks, eye_indices, iris_indices, img_w, img_h):
        eye_points = self.get_eye_points(landmarks, eye_indices, img_w, img_h)
        iris_points = self.get_eye_points(landmarks, iris_indices, img_w, img_h)
        
        eye_left = np.min(eye_points[:, 0])
        eye_right = np.max(eye_points[:, 0])
        eye_width = eye_right - eye_left
        
        iris_center_x = np.mean(iris_points[:, 0])
        relative_pos = (iris_center_x - eye_left) / eye_width - 0.5
        return relative_pos * 2
    
    def calculate_head_pose(self, landmarks, img_w, img_h):
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        nose = np.array([nose_tip.x * img_w, nose_tip.y * img_h])
        left = np.array([left_eye.x * img_w, left_eye.y * img_h])
        right = np.array([right_eye.x * img_w, right_eye.y * img_h])
        
        eye_center = (left + right) / 2
        horizontal_angle = np.arctan2(right[1] - left[1], right[0] - left[0])
        vertical_vec = nose - eye_center
        vertical_angle = np.arctan2(vertical_vec[0], vertical_vec[1])
        
        return np.degrees(horizontal_angle), np.degrees(vertical_angle)
    
    def calculate_attention_score(self, ear_left, ear_right, gaze_left, gaze_right, 
                                  head_angle_h, head_angle_v, blink_rate):
        score = 0
        
        avg_ear = (ear_left + ear_right) / 2
        if avg_ear > self.blink_threshold:
            eye_score = min(25, (avg_ear - self.blink_threshold) / 0.08 * 25)
        else:
            eye_score = 0
        score += eye_score
        
        avg_gaze = abs((gaze_left + gaze_right) / 2)
        if avg_gaze < self.GAZE_THRESHOLD:
            gaze_score = 45 * (1 - (avg_gaze / self.GAZE_THRESHOLD) ** 0.7)
        else:
            gaze_score = 0
        score += gaze_score
        
        head_deviation = np.sqrt(head_angle_h**2 + head_angle_v**2)
        if head_deviation < self.HEAD_POSE_THRESHOLD:
            head_score = 20 * (1 - (head_deviation / self.HEAD_POSE_THRESHOLD) ** 0.5)
        else:
            head_score = 0
        score += head_score
        
        if self.BLINK_RATE_MIN <= blink_rate <= self.BLINK_RATE_MAX:
            blink_score = 10
        elif blink_rate > self.BLINK_RATE_MAX:
            blink_score = 7
        else:
            blink_score = 5
        score += blink_score
        
        return int(score)
    
    def get_attention_color(self, attention_score):
        if attention_score >= 80:
            return (0, 255, 0), "HIGHLY FOCUSED"
        elif attention_score >= 65:
            return (0, 255, 0), "FOCUSED"
        elif attention_score >= 45:
            return (0, 165, 255), "MODERATE"
        else:
            return (0, 0, 255), "DISTRACTED"
    
    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        attention_score = 0
        status = "NO FACE"
        color = (128, 128, 128)
        avg_ear = 0
        avg_gaze = 0
        head_angle_h = 0
        alert_triggered = False
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                )
            )
            
            left_eye_points = self.get_eye_points(landmarks, self.LEFT_EYE, img_w, img_h)
            right_eye_points = self.get_eye_points(landmarks, self.RIGHT_EYE, img_w, img_h)
            ear_left = self.calculate_ear(left_eye_points)
            ear_right = self.calculate_ear(right_eye_points)
            
            avg_ear = (ear_left + ear_right) / 2
            if avg_ear < self.blink_threshold:
                current_time = time.time()
                if current_time - self.last_blink_time > 0.3:
                    self.blink_counter += 1
                    self.last_blink_time = current_time
            
            blink_rate = self.blink_counter / max(1, (time.time() - self.last_blink_time + 0.1))
            
            gaze_left = self.get_iris_position(landmarks, self.LEFT_EYE, 
                                              self.LEFT_IRIS, img_w, img_h)
            gaze_right = self.get_iris_position(landmarks, self.RIGHT_EYE, 
                                               self.RIGHT_IRIS, img_w, img_h)
            
            avg_gaze = abs((gaze_left + gaze_right) / 2)
            head_angle_h, head_angle_v = self.calculate_head_pose(landmarks, img_w, img_h)
            
            attention_score = self.calculate_attention_score(
                ear_left, ear_right, gaze_left, gaze_right,
                head_angle_h, head_angle_v, blink_rate
            )
            
            self.attention_history.append(attention_score)
            avg_attention = int(np.mean(self.attention_history))
            
            color, status = self.get_attention_color(avg_attention)
            
            if self.enable_alerts:
                alert_triggered = self.alert_system.check_attention(avg_attention)
                
                if self.alert_system.distraction_start_time is not None:
                    distraction_time = int(time.time() - self.alert_system.distraction_start_time)
                    cv2.putText(frame, f"Distracted for: {distraction_time}s", (10, 230),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if self.alert_system.is_alerting:
                    cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 0, 255), 20)
                    cv2.putText(frame, "FOCUS ALERT!", (img_w//2 - 150, img_h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            if self.enable_logging:
                self.log_data(avg_attention, status, avg_ear, avg_gaze, 
                            abs(head_angle_h), alert_triggered)
            
            cv2.putText(frame, f"Attention: {avg_attention}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Status: {status}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Blinks: {self.blink_counter}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Gaze: {avg_gaze:.2f}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Head: {abs(head_angle_h):.1f}°", (10, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.enable_alerts:
                cv2.putText(frame, f"Alerts: {self.alert_system.total_alerts}", (10, 260),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            if self.enable_logging:
                cv2.circle(frame, (img_w - 30, 30), 8, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (img_w - 70, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press '0' to Quit", (img_w - 200, img_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame, attention_score, status

def main():
    print("🧠 FocusSense - AI Attention Detection System")
    print("=" * 50)
    print("✨ ALL FEATURES ENABLED:")
    print("   ✅ Feature 1: Data Logging")
    print("   ✅ Feature 2: Smart Alerts")
    print("   ✅ Feature 3: Session Reports")
    print("=" * 50)
    
    focus_sense = FocusSense(enable_logging=True, enable_alerts=True)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("📊 Session started")
    print("🔔 Alert system active")
    print("📈 Report will be generated on exit")
    print("Press '0' to quit and generate report")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        processed_frame, attention, status = focus_sense.process_frame(frame)
        cv2.imshow('FocusSense - Attention Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("📊 SESSION SUMMARY")
    print("=" * 50)
    
    if focus_sense.enable_logging:
        print(f"✅ Log file saved: {focus_sense.log_filename}")
    
    elapsed = int(time.time() - time.mktime(focus_sense.session_start.timetuple()))
    mins = elapsed // 60
    secs = elapsed % 60
    print(f"⏱️  Session duration: {mins}m {secs}s")
    print(f"👁️  Total blinks: {focus_sense.blink_counter}")
    
    if len(focus_sense.attention_history) > 0:
        avg_attention = int(np.mean(focus_sense.attention_history))
        print(f"📈 Average attention: {avg_attention}%")
    
    if focus_sense.enable_alerts:
        print(f"🔔 Total alerts triggered: {focus_sense.alert_system.total_alerts}")
    
    print("=" * 50)
    
    # Generate visual report
    if focus_sense.enable_logging and MATPLOTLIB_AVAILABLE:
        print("\n🎨 Generating visual report...")
        report_gen = ReportGenerator(
            focus_sense.log_filename,
            focus_sense.session_start,
            focus_sense.alert_system
        )
        report_gen.generate_report()
        print("\n🌐 Report opened in your browser!")
    elif not MATPLOTLIB_AVAILABLE:
        print("\n💡 Install matplotlib to generate visual reports:")
        print("   pip install matplotlib")
    
    print("\n✅ FocusSense closed successfully")

if __name__ == "__main__":
    main()