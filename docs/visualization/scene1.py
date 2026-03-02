from manim import *
import pandas as pd
import numpy as np

DATA_DIR = '/Users/macbook/Downloads/HCMUT/Assignments/AI Projects/trend-spy-bot/src/backend/models/data'
START_DATE = "2021-06-01"
END_DATE = "2022-12-31"

class Scene1_RealOHLCV(MovingCameraScene):
    def construct(self):
        df_norm = self.load_data(START_DATE, END_DATE)
        
        if df_norm is None:
            return

        y_min = df_norm.min().min()
        y_max = df_norm.max().max()
        
        y_axis_max = np.ceil(y_max * 10) / 10 + 0.1
        y_axis_min = np.floor(y_min * 10) / 10 - 0.1
        
        num_days = len(df_norm)
        
        axes = Axes(
            x_range=[0, num_days, max(1, num_days // ((pd.to_datetime(END_DATE).year - pd.to_datetime(START_DATE).year) * 12 + pd.to_datetime(END_DATE).month - pd.to_datetime(START_DATE).month + 1))],
            y_range=[y_axis_min, y_axis_max, 0.1],
            x_length=11,
            y_length=7,
            axis_config={"include_tip": False, "color": GRAY},
            y_axis_config={
                "include_numbers": True, 
                "font_size": 17,
                "decimal_number_config": {"num_decimal_places": 1},
            }
        ).center()
        
        labels = axes.get_axis_labels(
            x_label=Text("Trading Days", font_size=22), 
            y_label=Text("Return (%)", font_size=20)
        )

        stock_config = {
            "SPY": {"color": WHITE, "dashed": True},
            "AAPL": {"color": BLUE, "dashed": False},
            "MSFT": {"color": GREEN, "dashed": False},
            "GOOGL": {"color": YELLOW, "dashed": False},
            "META": {"color": RED, "dashed": False}
        }

        lines = []
        text_labels = []
        final_tags = []
        
        line_anims = []
        label_anims = []
        tag_anims = []
        fadeout_tags = []

        for ticker, style in stock_config.items():
            points = [axes.c2p(i, row[ticker]) for i, row in df_norm.iterrows()]
            
            line = VMobject().set_points_as_corners(points).set_color(style["color"])
            if style["dashed"]:
                line = DashedVMobject(line, num_dashes=100)
            
            label = Text(ticker, font_size=18, color=style["color"]).next_to(points[-1], RIGHT)
            
            final_val = df_norm[ticker].iloc[-1]
            tag_text = f"{final_val*100:+.1f}\%"
            tag = MathTex(tag_text, color=style["color"], font_size=24).next_to(label, RIGHT)

            lines.append(line)
            text_labels.append(label)
            final_tags.append(tag)
            
            line_anims.append(Create(line))
            label_anims.append(FadeIn(label))
            tag_anims.append(Write(tag))
            fadeout_tags.append(FadeOut(tag))

        self.play(Create(axes), Write(labels))
        self.wait(0.7)
        
        self.play(
            *line_anims,
            run_time=3,
            rate_func=linear
        )
        
        self.play(FadeOut(labels), run_time=0.5)
        self.play(*label_anims)
        
        self.remove(labels)
        
        zoom_center = axes.c2p(num_days, (y_max + y_min)/2)
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(zoom_center),
            run_time=2
        )
        
        self.play(*tag_anims)
        
        self.wait(1)
        self.remove(*fadeout_tags)
        
        self.play(
            FadeOut(axes), 
            *[FadeOut(obj) for obj in lines + text_labels + final_tags],
            self.camera.frame.animate.scale(2).move_to(ORIGIN),
            run_time=1.5
        )
        
        final_stats = []
        for ticker, style in stock_config.items():
            final_stats.append({
                "name": ticker,
                "ret": df_norm[ticker].iloc[-1],
                "color": style["color"]
            })
        final_stats.sort(key=lambda x: x["ret"], reverse=True)
        
        title = Text("Relative Performance Ranking", font_size=40).to_edge(UP)
        self.play(Write(title), run_time=1)

        for i, item in enumerate(final_stats):
            rank = Text(f"#{i+1}", font_size=36, color=GRAY).shift(UP*(2 - i*1.0) + LEFT*3)
            name = Text(item["name"], font_size=36, color=item["color"]).next_to(rank, RIGHT, buff=1)
            val_text = f"{item['ret']*100:+.1f}%"
            val = Text(val_text, font_size=36, color=item["color"]).next_to(name, RIGHT, buff=1)

            anims = [FadeIn(rank), FadeIn(name), FadeIn(val)]

            if i == 0:
                winner_group = VGroup(rank, name, val)
                box = SurroundingRectangle(winner_group, color=YELLOW, buff=0.06)
                alpha_txt = Text("ALPHA LEADER", font_size=20, color=YELLOW).next_to(box, UP)
                anims.extend([Create(box), FadeIn(alpha_txt)])
            
            self.play(*anims, run_time=0.6)
        self.wait(1)

    def load_data(self, start, end):
        tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'META']
        
        dfs = []
        for t in tickers:
            file_path = f"{DATA_DIR}/{t}.csv"
            df = pd.read_csv(file_path, parse_dates=['Date'])
            
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
        
        return df_norm

class SMADistanceScene(MovingCameraScene):
    def construct(self):
        # 1. Load and Prep Data
        df = self.load_data(START_DATE, END_DATE)
        if df is None: return

        # Calculate SMAs
        df['SMA10'] = df['Price'].rolling(window=10).mean()
        df['SMA20'] = df['Price'].rolling(window=20).mean()
        df['SMA60'] = df['Price'].rolling(window=60).mean()
        
        # Crop to where valid data exists
        df = df.dropna().reset_index(drop=True)
        # Limit frames for smoother animation if dataset is huge
        if len(df) > 200: df = df.iloc[-200:].reset_index(drop=True)

        # 2. Set up Main Axes
        price_min = df[['Price', 'SMA10', 'SMA20', 'SMA60']].min().min() * 0.95
        price_max = df[['Price', 'SMA10', 'SMA20', 'SMA60']].max().max() * 1.05
        
        axes = Axes(
            x_range=[0, len(df), 30],
            y_range=[price_min, price_max, (price_max - price_min)/5],
            x_length=11,
            y_length=5,
            axis_config={"include_tip": False, "color": GRAY},
            y_axis_config={
                "include_numbers": True, 
                "font_size": 18,
                "decimal_number_config": {"num_decimal_places": 0}
            }
        ).shift(UP * 0.5)
        
        labels = axes.get_axis_labels(
            x_label=Text("Trading Days", font_size=20),
            y_label=Text("Price", font_size=20)
        )

        self.play(Create(axes), Write(labels))

        # 3. Plot Price Line
        price_line = self.get_line_graph(df, 'Price', axes, WHITE, stroke_width=2)
        price_label = Text("Price", font_size=20, color=WHITE).next_to(price_line.get_end(), RIGHT)
        
        self.play(Create(price_line), run_time=2, rate_func=linear)
        self.play(FadeIn(price_label))

        # 4. Animate SMAs
        sma_configs = [
            ('SMA10', YELLOW, "SMA(10)"),
            ('SMA20', ORANGE, "SMA(20)"),
            ('SMA60', RED, "SMA(60)")
        ]
        
        sma_lines = {}
        sma_labels = []

        for col, color, txt in sma_configs:
            line = self.get_line_graph(df, col, axes, color, stroke_width=2)
            lbl = Text(txt, font_size=18, color=color).next_to(line.get_end(), RIGHT)
            
            sma_lines[col] = line
            sma_labels.append(lbl)
            
            self.play(
                Create(line), 
                FadeIn(lbl, shift=LEFT), 
                run_time=1.5
            )

        # 5. Highlight "Distance from SMA" Concept
        # Pick a specific interesting point (e.g., index 120)
        idx = min(120, len(df)-10)
        
        # Zoom Camera
        zoom_point = axes.c2p(idx, df.iloc[idx]['Price'])
        self.play(
            self.camera.frame.animate.scale(0.4).move_to(zoom_point),
            run_time=2,
            rate_func=smooth
        )

        # Draw the Distance Line (Gap)
        price_pt = axes.c2p(idx, df.iloc[idx]['Price'])
        sma60_pt = axes.c2p(idx, df.iloc[idx]['SMA60'])
        
        # Create a visual vertical line connecting Price and SMA60
        gap_line = DashedLine(price_pt, sma60_pt, color=BLUE)
        gap_brace = Brace(gap_line, RIGHT, color=BLUE, buff=0.05)
        
        dist_text = MathTex(r"d_t", color=BLUE, font_size=32).next_to(gap_brace, RIGHT)
        
        self.play(Create(gap_line), Create(gap_brace), Write(dist_text))
        
        # Formula display (in zoomed view)
        formula = MathTex(
            r"d_t = \frac{P_t - \text{SMA}_{60}}{\text{SMA}_{60}}",
            font_size=28,
            color=BLUE
        ).next_to(dist_text, RIGHT, buff=0.3)
        
        bg_rect = BackgroundRectangle(formula, color=BLACK, fill_opacity=0.7)
        
        self.play(FadeIn(bg_rect), Write(formula))
        self.wait(1)

        # 6. Transition to Indicator View (Zoom Out)
        self.play(
            FadeOut(gap_line), FadeOut(gap_brace), FadeOut(dist_text), 
            FadeOut(bg_rect), FadeOut(formula),
            self.camera.frame.animate.scale(2.5).move_to(ORIGIN),
            run_time=1.5
        )

        # Create bottom panel for the Distance Oscillator
        dist_values = (df['Price'] - df['SMA60']) / df['SMA60']
        max_dist = max(dist_values.abs().max(), 0.05)
        
        dist_axes = Axes(
            x_range=[0, len(df), 30],
            y_range=[-max_dist, max_dist, max_dist],
            x_length=11,
            y_length=2,
            axis_config={"include_tip": False, "color": GRAY},
             y_axis_config={
                "include_numbers": True, 
                "decimal_number_config": {"num_decimal_places": 2},
                "font_size": 16
            }
        ).next_to(axes, DOWN, buff=0.7)
        
        dist_label = Text("Relative Distance (SMA 60)", font_size=20, color=BLUE).next_to(dist_axes, UP, aligned_edge=LEFT)
        zero_line = Line(dist_axes.c2p(0,0), dist_axes.c2p(len(df),0), color=WHITE, stroke_opacity=0.5)

        # Plot the oscillator curve
        dist_points = [dist_axes.c2p(i, v) for i, v in enumerate(dist_values)]
        dist_curve = VMobject().set_points_as_corners(dist_points).set_color(BLUE)

        # Area under curve (Green for positive, Red for negative) -- Simplified with simple lines for performance
        area_lines = VGroup()
        for i, val in enumerate(dist_values):
            if i % 2 == 0: # Optimization: skip every other point for rendering speed
                p1 = dist_axes.c2p(i, 0)
                p2 = dist_axes.c2p(i, val)
                color = GREEN if val > 0 else RED
                line = Line(p1, p2, color=color, stroke_width=1, stroke_opacity=0.6)
                area_lines.add(line)

        self.play(
            Create(dist_axes), 
            Write(dist_label), 
            Create(zero_line),
            run_time=1
        )
        
        self.play(
            Create(dist_curve),
            FadeIn(area_lines),
            run_time=2
        )
        
        self.wait(2)

    def get_line_graph(self, df, col_name, axes, color, stroke_width=2):
        points = [axes.c2p(i, row[col_name]) for i, row in df.iterrows()]
        line = VMobject().set_points_as_corners(points).set_color(color).set_stroke(width=stroke_width)
        return line

    def load_data(self, start, end):
        tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'META']
        
        dfs = []
        for t in tickers:
            file_path = f"{DATA_DIR}/{t}.csv"
            df = pd.read_csv(file_path, parse_dates=['Date'])
            
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
        
        return df_norm