from manim import *
import numpy as np
from scipy.stats import norm

class GaussianConfidenceScene(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-0.15, 0.15, 0.05],
            y_range=[0, 25, 5],
            x_length=10,
            y_length=5,
            axis_config={
                "color": GREY, 
                "stroke_width": 2, 
                "include_tip": False,
            },
            y_axis_config={
                "decimal_number_config": {"num_decimal_places": 2}
            }
        )
        axes.add_coordinates(font_size=16)
        axes.to_edge(DOWN, buff=1.0)
        
        x_label = Text("Predicted Return (5-Day)", font_size=20).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Probability Density", font_size=20).next_to(axes.y_axis, UP, buff=0.2).rotate(90*DEGREES)
        
        zero_line = Line(
            axes.c2p(0, 0), axes.c2p(0, 25), 
            color=WHITE, stroke_width=2, stroke_opacity=0.5
        )
        
        self.play(
            Create(axes), Write(x_label), Write(y_label), 
            Create(zero_line)
        )

        mu_tracker = ValueTracker(0.005) 
        sigma_tracker = ValueTracker(0.04) 

        def get_gaussian_curve():
            mu = mu_tracker.get_value()
            sigma = sigma_tracker.get_value()
            fn = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            curve = axes.plot(fn, x_range=[-0.15, 0.15], color=BLUE)
            return curve

        curve = always_redraw(get_gaussian_curve)
        self.add(curve)

        def get_positive_area():
            mu = mu_tracker.get_value()
            sigma = sigma_tracker.get_value()
            fn = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            temp_graph = axes.plot(fn, x_range=[-0.15, 0.15])
            area = axes.get_area(
                temp_graph, 
                x_range=[0, 0.15], 
                color=GREEN, 
                opacity=0.4
            )
            return area

        def get_negative_area():
            mu = mu_tracker.get_value()
            sigma = sigma_tracker.get_value()
            fn = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            temp_graph = axes.plot(fn, x_range=[-0.15, 0.15])
            area = axes.get_area(
                temp_graph, 
                x_range=[-0.15, 0], 
                color=RED, 
                opacity=0.4
            )
            return area

        pos_area = always_redraw(get_positive_area)
        neg_area = always_redraw(get_negative_area)
        self.add(neg_area, pos_area)

        mu_tex = MathTex(r"\mu =", font_size=36).set_color(BLUE)
        mu_num = DecimalNumber(0, num_decimal_places=3, include_sign=True, font_size=36).set_color(BLUE)
        mu_num.add_updater(lambda m: m.set_value(mu_tracker.get_value()))
        mu_group = VGroup(mu_tex, mu_num).arrange(RIGHT)
        
        sigma_tex = MathTex(r"\sigma =", font_size=36).set_color(YELLOW)
        sigma_num = DecimalNumber(0, num_decimal_places=3, font_size=36).set_color(YELLOW)
        sigma_num.add_updater(lambda m: m.set_value(sigma_tracker.get_value()))
        sigma_group = VGroup(sigma_tex, sigma_num).arrange(RIGHT)
        
        conf_tex = Text("Confidence P(R>0): ", font_size=24)
        conf_num = DecimalNumber(0, num_decimal_places=1, unit="%", font_size=20).set_color(GREEN)
        conf_num.next_to(conf_tex, RIGHT, buff=0.2)

        def update_conf_num(m):
            mu = mu_tracker.get_value()
            sigma = sigma_tracker.get_value()
            prob_loss = norm.cdf(0, loc=mu, scale=sigma)
            prob_win = 1 - prob_loss
            m.set_value(prob_win * 100)
            if prob_win > 0.52:
                m.set_color(GREEN)
            else:
                m.set_color(GREY)
            m.next_to(conf_tex, RIGHT, buff=0.2, aligned_edge=DOWN)

        conf_num.add_updater(update_conf_num)
        
        conf_group = VGroup(conf_tex, conf_num)
        
        stats_panel = VGroup(mu_group, sigma_group, conf_group).arrange(DOWN, aligned_edge=LEFT)
        stats_panel.to_corner(UR).shift(LEFT * 0.5)
        
        stats_bg = BackgroundRectangle(stats_panel, color=BLACK, fill_opacity=0.8, buff=0.2)
        
        self.play(FadeIn(stats_bg), FadeIn(stats_panel))

        title_a = Text("Scenario 1: High Uncertainty", font_size=28, color=GREY).to_corner(UL)
        self.play(Write(title_a))
        self.wait(1)
        
        self.play(
            mu_tracker.animate.set_value(-0.01),
            sigma_tracker.animate.set_value(0.06),
            run_time=2
        )
        self.wait(1)
        
        title_b = Text("Scenario 2: Alpha Detected (Strong Signal)", font_size=28, color=GREEN).to_corner(UL)
        
        self.play(
            Transform(title_a, title_b),
            mu_tracker.animate.set_value(0.04),
            sigma_tracker.animate.set_value(0.02),
            run_time=2
        )
        self.wait(1)
        
        threshold_line = Line(stats_panel.get_left(), stats_panel.get_right(), color=GREEN).next_to(conf_group, DOWN, buff=0.1)
        threshold_lbl = Text("Threshold > 52% -> BUY", font_size=18, color=GREEN).next_to(threshold_line, DOWN)
        
        self.play(Create(threshold_line), FadeIn(threshold_lbl))
        self.play(Indicate(conf_group, scale_factor=1.2))
        
        self.wait(2)