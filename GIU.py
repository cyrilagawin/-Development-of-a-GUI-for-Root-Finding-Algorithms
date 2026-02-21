import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class RootFinderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods GUI - Root Finding")
        
        # --- UI LAYOUT ---
        self.setup_input_frame()
        self.setup_output_frames()
        
    def setup_input_frame(self):
        # Top Frame for Inputs (Requirement 4, Step 2)
        input_frame = ttk.LabelFrame(self.root, text="Parameters")
        input_frame.pack(side="top", fill="x", padx=10, pady=10)
        
        # Function Input [cite: 21]
        ttk.Label(input_frame, text="f(x):").grid(row=0, column=0, sticky="w")
        self.equation_entry = ttk.Entry(input_frame, width=30)
        self.equation_entry.insert(0, "x**3 - x - 1")
        self.equation_entry.grid(row=0, column=1, padx=5)

        # Method Selection [cite: 22, 23]
        ttk.Label(input_frame, text="Method:").grid(row=0, column=2, sticky="w")
        self.method_var = tk.StringVar(value="Bisection")
        self.method_menu = ttk.Combobox(input_frame, textvariable=self.method_var, state="readonly")
        self.method_menu['values'] = ("Bisection", "False Position", "Newton-Raphson", "Secant")
        self.method_menu.grid(row=0, column=3, padx=5)
        self.method_menu.bind("<<ComboboxSelected>>", self.toggle_inputs)

        # Bounds and Tolerance [cite: 25, 26, 27]
        ttk.Label(input_frame, text="x_l / x_0:").grid(row=1, column=0)
        self.xl_entry = ttk.Entry(input_frame, width=10)
        self.xl_entry.insert(0, "1")
        self.xl_entry.grid(row=1, column=1, sticky="w")

        self.xu_label = ttk.Label(input_frame, text="x_u / x_1:")
        self.xu_label.grid(row=1, column=2)
        self.xu_entry = ttk.Entry(input_frame, width=10)
        self.xu_entry.insert(0, "2")
        self.xu_entry.grid(row=1, column=3, sticky="w")

        ttk.Label(input_frame, text="Tol:").grid(row=2, column=0)
        self.tol_entry = ttk.Entry(input_frame, width=10)
        self.tol_entry.insert(0, "0.0001")
        self.tol_entry.grid(row=2, column=1, sticky="w")

        # Calculate Button [cite: 48, 52]
        self.calc_btn = ttk.Button(input_frame, text="Calculate", command=self.calculate)
        self.calc_btn.grid(row=2, column=3, pady=5)

    def setup_output_frames(self):
        # Bottom-Left: Iteration Table [cite: 49]
        self.table_frame = ttk.Frame(self.root)
        self.table_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        columns = ("Iter", "Approximation", "f(x)", "Error")
        self.tree = ttk.Treeview(self.table_frame, columns=columns, show="headings", height=10)
        for col in columns: 
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        self.tree.pack(fill="both", expand=True)

        # Bottom-Right: Interactive Graph [cite: 50, 33]
        graph_container = ttk.Frame(self.root)
        graph_container.pack(side="right", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        
        # Navigation Toolbar (Supports zoom to rectangle) [cite: 11]
        self.toolbar = NavigationToolbar2Tk(self.canvas, graph_container)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        # Bind Mouse Wheel for "Pinch-style" Zooming
        self.canvas.mpl_connect("scroll_event", self.zoom_fun)

    def zoom_fun(self, event):
        # Determine the zoom scale factor
        base_scale = 1.1
        if event.button == 'up': # Zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down': # Zoom out
            scale_factor = base_scale
        else:
            scale_factor = 1

        # Get the current limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        # Set new limits centered on mouse position
        rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * (rel_x)])
        self.ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * (rel_y)])
        self.canvas.draw()

    def toggle_inputs(self, event=None):
        # Requirement: Disable x_u for Newton-Raphson [cite: 26]
        if self.method_var.get() == "Newton-Raphson":
            self.xu_entry.config(state="disabled")
        else:
            self.xu_entry.config(state="normal")

    def evaluate_function(self, x_val):
        # Safely evaluate user string (Step 1) [cite: 40, 41]
        expr = self.equation_entry.get().replace('^', '**')
        safe_dict = {"x": x_val, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp}
        return eval(expr, {"__builtins__": None}, safe_dict)

    def calculate(self):
        try:
            method = self.method_var.get()
            x0 = float(self.xl_entry.get())
            tol = float(self.tol_entry.get())
            max_iter = 100 # Safety stop [cite: 29]
            
            if method == "Newton-Raphson":
                root_val, history = self.newton_raphson(x0, tol, max_iter)
            else:
                x1 = float(self.xu_entry.get())
                if method == "Bisection":
                    root_val, history = self.bisection(x0, x1, tol, max_iter)
                elif method == "False Position":
                    root_val, history = self.false_position(x0, x1, tol, max_iter)
                elif method == "Secant":
                    root_val, history = self.secant(x0, x1, tol, max_iter)

            self.update_ui(root_val, history)
        except Exception as e:
            messagebox.showerror("Input Error", f"Check your function or parameters.\n{e}")

    # --- ALGORITHMS (Requirement 4, Step 1) ---
    def bisection(self, xl, xu, tol, max_iter):
        history = []
        if self.evaluate_function(xl) * self.evaluate_function(xu) >= 0:
            raise ValueError("f(xl) and f(xu) must have opposite signs.")
        for i in range(1, max_iter + 1):
            xr = (xl + xu) / 2
            fxr = self.evaluate_function(xr)
            error = abs(xu - xl) / 2
            history.append([i, f"{xr:.6f}", f"{fxr:.6e}", f"{error:.6e}"])
            if error < tol: break
            if self.evaluate_function(xl) * fxr < 0: xu = xr
            else: xl = xr
        return xr, history

    def false_position(self, xl, xu, tol, max_iter):
        history = []
        xr_old = xl
        for i in range(1, max_iter + 1):
            fl, fu = self.evaluate_function(xl), self.evaluate_function(xu)
            xr = xu - (fu * (xl - xu)) / (fl - fu)
            fxr = self.evaluate_function(xr)
            error = abs(xr - xr_old)
            history.append([i, f"{xr:.6f}", f"{fxr:.6e}", f"{error:.6e}"])
            if error < tol and i > 1: break
            if fl * fxr < 0: xu = xr
            else: xl = xr
            xr_old = xr
        return xr, history

    def newton_raphson(self, x0, tol, max_iter):
        history = []
        x = x0
        for i in range(1, max_iter + 1):
            fx = self.evaluate_function(x)
            h = 1e-8
            dfx = (self.evaluate_function(x + h) - fx) / h # Numerical derivative
            if dfx == 0: break
            x_new = x - fx / dfx
            error = abs(x_new - x)
            history.append([i, f"{x_new:.6f}", f"{fx:.6e}", f"{error:.6e}"])
            if error < tol: break
            x = x_new
        return x, history

    def secant(self, x0, x1, tol, max_iter):
        history = []
        for i in range(1, max_iter + 1):
            f0, f1 = self.evaluate_function(x0), self.evaluate_function(x1)
            if f1 - f0 == 0: break
            x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
            error = abs(x_new - x1)
            history.append([i, f"{x_new:.6f}", f"{f1:.6e}", f"{error:.6e}"])
            if error < tol: break
            x0, x1 = x1, x_new
        return x1, history

    def update_ui(self, root_val, history):
        # Update Table [cite: 56]
        for row in self.tree.get_children(): self.tree.delete(row)
        for row in history: self.tree.insert("", "end", values=row)

        # Update Graph [cite: 57, 58, 59, 60, 62]
        self.ax.clear()
        x_range = np.linspace(root_val - 2, root_val + 2, 500)
        y_vals = [self.evaluate_function(v) for v in x_range]
        self.ax.plot(x_range, y_vals, label='f(x)')
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.axvline(0, color='black', linewidth=1)
        self.ax.scatter([root_val], [0], color='red', zorder=5, label=f'Root ≈ {root_val:.4f}')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    def on_closing():
        root.destroy()
        root.quit()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = RootFinderGUI(root)
    root.mainloop()
