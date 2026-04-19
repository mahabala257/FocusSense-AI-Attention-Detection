"""
Quick script to open the latest FocusSense report
"""
import os
import glob

def view_latest_report():
    logs_dir = 'logs'
    
    if not os.path.exists(logs_dir):
        print("❌ No logs folder found!")
        return
    
    # Find all HTML reports
    html_files = glob.glob(f'{logs_dir}/report_*.html')
    
    if not html_files:
        print("❌ No reports found!")
        print("💡 Run focussense.py first to generate reports")
        return
    
    # Get the latest HTML file
    latest_html = max(html_files, key=os.path.getmtime)
    latest_png = latest_html.replace('.html', '.png')
    
    print(f"📊 Latest Report: {latest_html}")
    print(f"📈 Image Report: {latest_png}")
    print("\n🌐 Opening reports...")
    
    # Open both HTML and PNG
    try:
        abs_html = os.path.abspath(latest_html)
        abs_png = os.path.abspath(latest_png)
        
        # Open PNG (always works)
        if os.path.exists(abs_png):
            os.startfile(abs_png)
            print(f"✅ Opened: {abs_png}")
        
        # Try to open HTML
        if os.path.exists(abs_html):
            os.startfile(abs_html)
            print(f"✅ Opened: {abs_html}")
            
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print(f"\n📂 Open manually from:")
        print(f"   {os.path.abspath(logs_dir)}")

if __name__ == "__main__":
    print("🎨 FocusSense Report Viewer")
    print("=" * 50)
    view_latest_report()
    print("\n✅ Done!")