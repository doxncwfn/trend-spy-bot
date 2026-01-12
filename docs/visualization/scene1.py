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
        
        try:
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

        except Exception as e:
            print(f"Data Loading Error: {e}")
            print("Generating DUMMY data for visualization purposes...")
            dates = pd.date_range(start=start, end=end, freq='D')
            
            data = {}
            data['SPY'] = np.cumsum(np.random.normal(0.0003, 0.01, size=len(dates)))
            data['AAPL'] = np.cumsum(np.random.normal(0.0005, 0.015, size=len(dates)))
            data['GOOGL'] = np.cumsum(np.random.normal(0.0004, 0.014, size=len(dates)))
            data['META'] = np.cumsum(np.random.normal(0.0006, 0.020, size=len(dates)))
            
            return pd.DataFrame(data, index=dates).reset_index(drop=True)