from manim import *
import pandas as pd
import numpy as np

DATA_DIR = '/Users/macbook/Downloads/HCMUT/Assignments/AI Projects/trend-spy-bot/src/backend/models/data'
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

class Scene2_SecretSauceReal(Scene):
    def construct(self):
        day_data, date_str = self.get_worst_day_data()
        if day_data is None: return

        tickers = list(day_data.keys())
        raw_values = np.array(list(day_data.values()))
        
        if np.mean(np.abs(raw_values)) > 1.0:
            print("Detected values > 1.0, assuming percentage integers. Dividing by 100.")
            raw_values = raw_values / 100.0

        mean_ret = np.mean(raw_values)
        
        data_min = raw_values.min()
        data_max = raw_values.max()
        
        y_min_limit = min(data_min, -0.05) * 1.3
        y_max_limit = max(data_max, 0.05) * 1.3

        y_min_limit = np.floor(y_min_limit * 100) / 100
        y_max_limit = np.ceil(y_max_limit * 100) / 100

        raw_range = y_max_limit - y_min_limit
        if raw_range > 0.2:
            tick_step = 0.05
        elif raw_range > 0.1:
            tick_step = 0.02
        else:
            tick_step = 0.01

        axes = Axes(
            x_range=[0, len(tickers) + 1, 1],
            y_range=[y_min_limit, y_max_limit, tick_step],
            x_length=10,
            y_length=5.0,
            axis_config={
                "include_tip": False,
                "color": GRAY
            },
            y_axis_config={
                "include_numbers": True,
                "decimal_number_config": {
                    "num_decimal_places": 2
                }
            }
        ).center().shift(DOWN * 0.3)
        
        zero_line = Line(axes.c2p(0, 0), axes.c2p(len(tickers)+1, 0), color=WHITE, stroke_width=2)
        y_lbl = axes.get_y_axis_label(Text("Return", font_size=36)).scale(0.7).next_to(axes.y_axis, UP)

        title = Text(f"Market Crash Analysis: {date_str}", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        bars = VGroup()
        labels = VGroup()
        val_texts = VGroup()
        BAR_WIDTH = 0.8

        for i, val in enumerate(raw_values):
            p_origin = axes.c2p(i+1, 0)
            p_val = axes.c2p(i+1, val)
            bar_height = abs(p_val[1] - p_origin[1])
            
            bar = Rectangle(
                height=bar_height,
                width=BAR_WIDTH,
                fill_color=RED,
                fill_opacity=0.9,
                stroke_width=0
            )
            
            if val < 0:
                bar.move_to(p_origin, aligned_edge=UP)
                lbl = Text(tickers[i], font_size=20).next_to(p_origin, UP, buff=0.2)
                vt = Text(f"{val*100:.1f}%", font_size=18, color=RED).next_to(bar, DOWN, buff=0.1)
            else:
                bar.move_to(p_origin, aligned_edge=DOWN)
                lbl = Text(tickers[i], font_size=20).next_to(p_origin, DOWN, buff=0.2)
                vt = Text(f"{val*100:.1f}%", font_size=18, color=RED).next_to(bar, UP, buff=0.1)

            bars.add(bar)
            labels.add(lbl)
            val_texts.add(vt)

        self.play(Create(axes), Create(zero_line), FadeIn(y_lbl))
        self.play(
            LaggedStart(*[GrowFromEdge(b, UP) for b in bars], lag_ratio=0.25),
            FadeIn(labels),
            FadeIn(val_texts),
            run_time=2
        )
        self.wait(1)

        mean_y = axes.c2p(0, mean_ret)[1]
        mean_line = DashedLine(
            start=[axes.c2p(0, 0)[0], mean_y, 0],
            end=[axes.c2p(len(tickers)+1, 0)[0], mean_y, 0],
            color=YELLOW,
            stroke_width=3
        )
        
        mean_text_str = f"Mean: {mean_ret*100:.1f}%"
        mean_lbl = Text(mean_text_str, color=YELLOW, font_size=24)
        mean_lbl_bg = BackgroundRectangle(mean_lbl, color=BLACK, fill_opacity=0.7, buff=0.1)
        mean_group = VGroup(mean_lbl_bg, mean_lbl).next_to(mean_line, RIGHT, aligned_edge=RIGHT)
        
        self.play(Create(mean_line), FadeIn(mean_group))
        self.wait(1)

        formula = MathTex(
            r"z_{i,t} = \frac{x_{i,t} - \mu_t}{\sigma_t}",
            font_size=40
        ).to_edge(UP).shift(DOWN*0.5)
        formula.set_color_by_tex("x_{i,t}", RED)
        formula.set_color_by_tex("\\mu_t", YELLOW)
        
        self.play(FadeOut(title), FadeIn(formula))
        
        shifted_values = raw_values - mean_ret 
        
        new_bars = VGroup()
        new_val_texts = VGroup()
        new_labels = []

        for i, val in enumerate(shifted_values):
            p_origin = axes.c2p(i+1, 0)
            p_val = axes.c2p(i+1, val)
            bar_height = abs(p_val[1] - p_origin[1])
            
            new_bar = Rectangle(
                height=bar_height,
                width=BAR_WIDTH,
                fill_color=GREEN if val >= 0 else RED, 
                fill_opacity=0.9,
                stroke_width=0
            )
            
            if val >= 0:
                new_bar.move_to(p_origin, aligned_edge=DOWN)
                nvt_pos = UP
                lbl_pos = DOWN
            else:
                new_bar.move_to(p_origin, aligned_edge=UP)
                nvt_pos = DOWN
                lbl_pos = UP
            
            # Name label
            new_lbl = Text(tickers[i], font_size=20)
            new_lbl.next_to(new_bar, lbl_pos, buff=0.2)
            new_labels.append(new_lbl)

            new_bars.add(new_bar)
            nvt_str = f"{val*100:+.1f}%"
            nvt_color = RED if val < 0 else GREEN
            nvt = Text(nvt_str, font_size=18, color=nvt_color)
            nvt.next_to(new_bar, nvt_pos, buff=0.1)
            new_val_texts.add(nvt)

        self.play(
            Indicate(formula),
            FadeOut(y_lbl),
            AnimationGroup(
                mean_line.animate.move_to(zero_line.get_center()),
                mean_group.animate.next_to(zero_line, RIGHT, buff=0.1).set_opacity(0),
                lag_ratio=0,
            ),
            Transform(bars, new_bars),
            FadeOut(val_texts),
            ReplacementTransform(labels, VGroup(*new_labels)),
            run_time=3
        )
        self.play(FadeIn(new_val_texts))
        
        winner_idx = np.argmax(shifted_values)
        loser_idx = np.argmin(shifted_values)
        
        win_group = VGroup(bars[winner_idx], new_val_texts[winner_idx])
        lose_group = VGroup(bars[loser_idx], new_val_texts[loser_idx])
        
        brace_win = Brace(win_group, UP, color=GREEN, buff=0.1)
        lbl_win = brace_win.get_text("Alpha").set_color(GREEN).scale(0.8)
        
        brace_lose = Brace(lose_group, DOWN, color=RED, buff=0.1)
        lbl_lose = brace_lose.get_text("Beta Drag").set_color(RED).scale(0.8)
        
        self.play(
            GrowFromCenter(brace_win), FadeIn(lbl_win),
            GrowFromCenter(brace_lose), FadeIn(lbl_lose),
            formula.animate.to_edge(DOWN, buff=0.6)
        )
        self.play(
            Create(SurroundingRectangle(formula, color=YELLOW, buff=0.15)),
            Indicate(formula),
            run_time = 1.5)
        
        self.wait(0.5)

    def get_worst_day_data(self):
        tickers = ['SPY', 'AAPL', 'NVDA', 'GOOGL', 'META']
        try:
            dfs = []
            for t in tickers:
                df = pd.read_csv(f"{DATA_DIR}/{t}.csv", parse_dates=['Date'])
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                df = df[['Date', price_col]].rename(columns={price_col: t})
                dfs.append(df)
            
            df_merged = dfs[0]
            for i in range(1, len(dfs)):
                df_merged = df_merged.merge(dfs[i], on='Date', how='inner')
            
            mask = (df_merged['Date'] >= START_DATE) & (df_merged['Date'] <= END_DATE)
            df_filtered = df_merged.loc[mask].set_index('Date')
            df_rets = df_filtered.pct_change().dropna()
            
            df_rets['mean'] = df_rets.mean(axis=1)
            worst_date = df_rets['mean'].idxmin()
            worst_row = df_rets.loc[worst_date].drop('mean')
            
            print(f"Worst Day: {worst_date.date()} ({df_rets.loc[worst_date]['mean']:.2%})")
            return worst_row.to_dict(), str(worst_date.date())
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None