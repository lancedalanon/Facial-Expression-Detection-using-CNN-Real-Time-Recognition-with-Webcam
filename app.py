import tkinter as tk
from page1 import FacialExpressionDetection
from page2 import PlaceholderPage

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Maximize window on startup
        self.state('zoomed')  # On macOS, consider fullscreen with `self.attributes("-fullscreen", True)`

        self.current_page = None
        self.show_page_1()

    def show_page_1(self):
        """Shows the real-time facial expression detection page."""
        if self.current_page:
            self.current_page.destroy()  # Remove the current page

        self.current_page = FacialExpressionDetection(self, self.show_page_2)
        self.current_page.pack(fill='both', expand=True)  # Make the page fill the window

    def show_page_2(self):
        """Shows a placeholder second page."""
        if self.current_page:
            self.current_page.destroy()

        self.current_page = PlaceholderPage(self, self.show_page_1)
        self.current_page.pack(fill='both', expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
