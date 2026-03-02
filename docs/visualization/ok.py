from manim import *
import pandas as pd
import numpy as np

DATA_DIR = '/Users/macbook/Downloads/HCMUT/Assignments/AI Projects/trend-spy-bot/src/backend/models/data'
START_DATE = "2022-01-01"
END_DATE = "2022-12-31"

def load_data(start, end):
    tickers = ['NVDA']
    try:
        dfs = []
        for t in tickers:
            file_path = f"{DATA_DIR}/{t}.csv"
            df = pd.read_csv(file_path, parse_dates=['Date'])
            if df.empty:
                raise ValueError(f"CSV file {file_path} is empty.")
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df = df[['Date', price_col]].rename(columns={price_col: t})
            dfs.append(df)

        df_merged = dfs[0]
        for i in range(1, len(dfs)):
            df_merged = df_merged.merge(dfs[i], on='Date', how='inner')

        mask = (df_merged['Date'] >= start) & (df_merged['Date'] <= end)
        df_final = df_merged.loc[mask].copy()

        if df_final.empty:
            print(f"Error: No data found between {start} and {end}.")
            return None

        df_final = df_final.reset_index(drop=True)
        cols = tickers
        initial_prices = df_final.iloc[0][cols]
        df_norm = df_final[cols] / initial_prices - 1
        print(f"Successfully loaded {tickers[0]} data.")
        return df_norm

    except Exception as e:
        print(f"Data Loading Error: {e}")
        print("Generating DUMMY data for visualization purposes...")
        dates = pd.date_range(start=start, end=end, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, size=len(dates))
        price_path = np.cumprod(1 + returns) - 1
        data = pd.DataFrame({'Date': dates, 'NVDA': price_path})
        return data
    
class SMADistanceScene(MovingCameraScene):
    def construct(self):
        # Set up
        df_norm = load_data(START_DATE, END_DATE)
        if df_norm is None:
            return

        df = df_norm.rename(columns={'NVDA': 'Price'})
        df['SMA10'] = df['Price'].rolling(window=10).mean()
        df['SMA20'] = df['Price'].rolling(window=20).mean()
        df['SMA60'] = df['Price'].rolling(window=60).mean()

        df = df.dropna().reset_index(drop=True)
        if len(df) > 300:
            df = df.iloc[-300:].reset_index(drop=True)

        vals = df[['Price', 'SMA10', 'SMA20', 'SMA60']]
        price_min = vals.min().min()
        price_max = vals.max().max()

        y_padding = (price_max - price_min) * 0.1
        price_min -= y_padding
        price_max += y_padding
        step_y = (price_max - price_min) / 4

        # ManimCE Axes uses x_length/y_length
        axes = Axes(
            x_range=[0, len(df), 30],
            y_range=[price_min, price_max, step_y],
            x_length=12,
            y_length=6,
            axis_config={
                "include_tip": False,
                "color": GREY,
                "stroke_width": 3
            },
            y_axis_config={
                "decimal_number_config": {"num_decimal_places": 2}
            }
        ).center()

        # ManimCE method to add numbers
        axes.add_coordinates(font_size=18)

        x_label = Text("Trading Days", font_size=20).next_to(axes.x_axis, UP, buff=0.2)
        y_label = Text("Norm. Return", font_size=20).next_to(axes.y_axis, UP, buff=0.2)
        labels = VGroup(x_label, y_label)

        self.play(Create(axes), Write(labels))
        
        # Price & SMAs
        price_line = self.get_line_graph(df, 'Price', axes, WHITE, stroke_width=3)
        price_label = Text("NVDA", font_size=20, color=WHITE).next_to(price_line.get_end(), RIGHT)

        self.play(Create(price_line), run_time=2, rate_func=linear)
        self.play(FadeIn(price_label))
        self.wait(0.5)
        self.play(FadeOut(price_label))

        sma_configs = [
            ('SMA10', GREEN, r"\text{SMA}_{10}"),
            ('SMA20', YELLOW, r"\text{SMA}_{20}"),
            ('SMA60', RED, r"\text{SMA}_{60}")
        ]
        sma_lines_group = VGroup()
        sma_labels_group = VGroup()
        for col, color, txt in sma_configs:
            line = self.get_line_graph(df, col, axes, color, stroke_width=4)
            lbl = MathTex(txt, font_size=24, color=color).next_to(line.get_end(), RIGHT)
            sma_lines_group.add(line)
            sma_labels_group.add(lbl)
            self.play(
                Create(line),
                FadeIn(lbl, shift=LEFT),
                run_time=1.5
            )
            self.wait(0.5)
            self.play(FadeOut(lbl))

        self.wait(2)
            
        # Distance to SMA_60
        idx = 124
        frame = self.camera.frame
        
        price_pt = axes.c2p(idx, df.iloc[idx]['Price'])
        sma60_pt = axes.c2p(idx, df.iloc[idx]['SMA60'])
        gap_line = DashedLine(price_pt, sma60_pt, color=BLUE)
        gap_brace = Brace(gap_line, RIGHT, buff=0.05)
        gap_brace.set_color(BLUE)

        self.play(
            frame.animate.scale(0.4).move_to(gap_line),
            run_time=2,
            rate_func=smooth
        )

        self.play(Create(gap_line), Create(gap_brace))

        formula = MathTex(
            r"d_{i,t} = P_t - \text{SMA}_{i,t}",
            font_size=24,
            color=BLUE
        ).next_to(gap_brace, RIGHT, buff=0.1)

        bg_rect = BackgroundRectangle(formula, color=BLACK, fill_opacity=0.7)

        self.play(FadeIn(bg_rect), Write(formula))
        self.wait(1)

        main_chart_group = VGroup(
            axes, labels, price_line, sma_lines_group
        )

        self.play(
            FadeOut(gap_line), FadeOut(gap_brace),
            FadeOut(bg_rect), FadeOut(formula),
            run_time=1
        )

        # Oscillator
        self.play(
            frame.animate.scale(2.5).move_to(ORIGIN),
            main_chart_group.animate.scale(0.5).to_edge(UP, buff=0.8),
            run_time=2,
            rate_func=smooth
        )

        dist_values = (df['Price'] - df['SMA60'])
        max_dist = max(dist_values.abs().max(), 0.01)

        dist_axes = Axes(
            x_range=[0, len(df), 30],
            y_range=[-max_dist, max_dist, max_dist],
            x_length=10,
            y_length=2.5,
            axis_config={"include_tip": False, "color": GREY},
            y_axis_config={"decimal_number_config": {"num_decimal_places": 2}}
        )
        dist_axes.add_coordinates(font_size=14)
        dist_axes.to_edge(DOWN, buff=0.5)

        dist_label = Text("Distance Feature (Oscillator)", font_size=20, color=BLUE)
        dist_label.next_to(dist_axes, UP, aligned_edge=LEFT)

        zero_line = Line(dist_axes.c2p(0, 0), dist_axes.c2p(len(df), 0), color=WHITE, stroke_opacity=0.5)

        dist_points = [dist_axes.c2p(i, v) for i, v in enumerate(dist_values)]
        dist_curve = VMobject().set_points_as_corners(dist_points).set_color(BLUE)

        area_lines = VGroup()
        for i, val in enumerate(dist_values):
            if i % 2 == 0:
                p1 = dist_axes.c2p(i, 0)
                p2 = dist_axes.c2p(i, val)
                color = GREEN if val > 0 else RED
                line = Line(p1, p2, color=color, stroke_width=1.5, stroke_opacity=0.6)
                area_lines.add(line)

        self.play(
            Create(dist_axes),
            Write(dist_label),
            Create(zero_line),
            run_time=2
        )

        self.play(
            Create(dist_curve),
            FadeIn(area_lines),
            run_time=3
        )
        self.wait(2)
        
    def get_line_graph(self, df, col_name, axes, color, stroke_width=2):
        points = [axes.c2p(i, row[col_name]) for i, row in df.iterrows()]
        line = VMobject().set_points_as_corners(points).set_color(color).set_stroke(width=stroke_width)
        return line