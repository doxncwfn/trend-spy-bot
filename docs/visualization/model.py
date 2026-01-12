from manim import *
import numpy as np
import random

# --- Configuration & Style ---
COLOR_INPUT = "#ADD8E6"
COLOR_LSTM = "#9370DB"
COLOR_EMBED = "#FFA500"
COLOR_TEXT = WHITE
COLOR_DROPOUT = "#444444"
COLOR_PACKET = "#FFFF00"

class LSTM(Scene):
    def construct(self):
        self.play_flattening_logic()
        self.play_lstm_mechanics()
        self.play_parallel_processing()

    def play_flattening_logic(self):
        """
        Visualizes the tensor reshaping: [B, S, T, F] -> [B*S, T, F]
        """
        stocks = VGroup()
        N = 53
        stock_labels = [f"Stock {i+1}" for i in range(5)] + ["...", f"Stock {N}"]
        for idx, stock_name in enumerate(stock_labels):
            rect = Rectangle(height=0.8, width=4, color=COLOR_INPUT, fill_opacity=0.4)
            side = Rectangle(height=0.8, width=0.2, color=BLUE_E, fill_opacity=0.6).next_to(rect, RIGHT, buff=0)
            top = Polygon(rect.get_corner(UL), rect.get_corner(UR), 
                          rect.get_corner(UR) + UP*0.2 + RIGHT*0.2, 
                          rect.get_corner(UL) + UP*0.2 + RIGHT*0.2,
                          color=BLUE_E, fill_opacity=0.6)
            if stock_name == "...":
                label = Text("...", font_size=26).move_to(rect)
            else:
                label = Text(stock_name, font_size=20).move_to(rect)
            block = VGroup(rect, side, top, label)
            block.shift(UP * 1 + DOWN * idx * 0.5 + RIGHT * idx * 0.3)
            stocks.add(block)
        stocks.move_to(ORIGIN)
        dims_text = MathTex(
            r"\text{Input: } [B=1, \mathbf{S=53}, T=60, F=9]", 
            tex_to_color_map={"S=53": COLOR_INPUT}
        ).next_to(stocks, UP, buff=0.5)
        self.play(FadeIn(stocks, lag_ratio=0.1), Write(dims_text))
        self.wait(1)

        # Move stack to a vertical format (batch processing)
        target_group = stocks.copy()
        target_group.arrange(DOWN, buff=0.15).to_edge(LEFT, buff=2)
        for block in target_group:
            block.remove(block[1], block[2])
            block[0].set_width(3)
        flatten_eq = MathTex(
            r"x_{flat} = x.\text{view}(B \times S, T, F)", 
            color=BLUE, font_size=36
        ).next_to(target_group, RIGHT, buff=1)
        dims_flat = MathTex(
            r"\text{New Shape: } [\mathbf{53}, 60, 9]", 
            font_size=32, color=GREY_B
        ).next_to(flatten_eq, DOWN)
        self.play(
            Transform(stocks, target_group),
            Transform(dims_text, flatten_eq),
            FadeIn(dims_flat),
            run_time=2
        )
        self.wait(1)

        self.flatten_visuals = VGroup(stocks, dims_text, dims_flat)
        self.hero_stock = stocks[0].copy()
        
        self.play(
            FadeOut(stocks[0:]),
            FadeOut(dims_text), 
            FadeOut(dims_flat),
            self.hero_stock.animate.move_to(UP * 3).scale(1.2),
            run_time=1.5
        )
        self.remove(stocks)

    def play_lstm_mechanics(self):
        """
        Visualizes the LSTM chain: input expansion, recurrence, dropout, and output highlighting.
        """
        num_visible_steps = 6
        inputs = VGroup()
        for i in range(num_visible_steps):
            dot = Circle(radius=0.25, color=COLOR_INPUT, fill_opacity=0.5, stroke_width=2)
            label = MathTex(f"x_{{{i+1}}}", font_size=23).move_to(dot)
            group = VGroup(dot, label)
            inputs.add(group)
        inputs.arrange(RIGHT, buff=1.2).shift(DOWN * 2)
        distribute_arrow = Arrow(self.hero_stock.get_bottom(), inputs.get_top(), color=RED_A)
        self.play(Create(distribute_arrow))
        self.play(
            LaggedStart(*[GrowFromPoint(inp, self.hero_stock.get_center()) for inp in inputs], lag_ratio=0.1),
            run_time=1.5
        )
        self.play(FadeOut(distribute_arrow), FadeOut(self.hero_stock))

        lstm_cells = VGroup()
        arrows_recurrence = VGroup()
        arrows_input = VGroup()
        cell_height = 1.2
        for i in range(num_visible_steps):
            cell = RoundedRectangle(corner_radius=0.2, height=cell_height, width=1.2, color=COLOR_LSTM, fill_opacity=0.2)
            cell.move_to(inputs[i].get_center() + UP * 2)
            lstm_cells.add(cell)
            arr_in = Arrow(inputs[i].get_top(), cell.get_bottom(), buff=0.1, color=COLOR_INPUT, max_tip_length_to_length_ratio=0.15)
            arrows_input.add(arr_in)
            if i > 0:
                arr_rec = Arrow(
                    lstm_cells[i-1].get_right(), 
                    cell.get_left(), 
                    buff=0.1, 
                    color=COLOR_LSTM,
                    max_tip_length_to_length_ratio=0.15
                )
                arrows_recurrence.add(arr_rec)

        # Step-by-step processing through the unrolled LSTM chain
        eq_lstm = MathTex(
            r"h_t = \text{LSTM}(h_{t-1}, x_t)", 
            font_size=36, 
            color=COLOR_LSTM
        ).to_edge(UP)
        self.play(Write(eq_lstm))
        for i in range(num_visible_steps):
            anims = []
            anims.append(Create(arrows_input[i]))
            packet_in = Dot(radius=0.08, color=COLOR_PACKET).move_to(inputs[i].get_center())
            anims.append(packet_in.animate.move_to(lstm_cells[i].get_center()))
            if i > 0:
                anims.append(Create(arrows_recurrence[i-1]))
                packet_rec = Dot(radius=0.08, color=COLOR_PACKET).move_to(lstm_cells[i-1].get_center())
                anims.append(packet_rec.animate.move_to(lstm_cells[i].get_center()))
            anims.append(GrowFromCenter(lstm_cells[i]))
            self.play(AnimationGroup(*anims), run_time=0.5)
            self.remove(packet_in)
            if i > 0: self.remove(packet_rec)

        # Dropout effect, neurons randomly "fizzing out"
        dropout_dots = VGroup()
        for cell in lstm_cells:
            neurons = VGroup(*[Dot(radius=0.035, color=WHITE) for _ in range(49)])
            neurons.arrange_in_grid(rows=7, cols=7, buff=0.07).move_to(cell)
            dropout_dots.add(neurons)
        self.play(FadeIn(dropout_dots))
        dropout_label = Text("Dropout 44%", font_size=20).next_to(lstm_cells, UP)
        self.play(FadeIn(dropout_label))
        self.wait(0.5)
        fizz_anims = []
        for neurons in dropout_dots:
            for neuron in neurons:
                if random.random() < 0.44:
                    fizz_anims.append(
                        AnimationGroup(
                            neuron.animate.set_color(RED).scale(1.5),
                            neuron.animate.set_opacity(0).scale(0),
                            lag_ratio=0.1
                        )
                    )
        self.play(AnimationGroup(*fizz_anims, lag_ratio=0.05), run_time=2)
        self.wait(0.5)
        self.play(FadeOut(dropout_label))

        # Last hidden state output
        last_cell = lstm_cells[-1]
        output_vec = VGroup(
            RoundedRectangle(height=2, width=0.4, corner_radius=0.1, color=COLOR_EMBED, fill_opacity=0.8),
            MathTex(r"h_{60}", color=WHITE, font_size=24)
        )
        output_vec[1].move_to(output_vec[0])
        output_vec.next_to(last_cell, RIGHT, buff=1.5)
        output_label = MathTex(r"h_{60} \in \mathbb{R}^{128}", font_size=24).next_to(output_vec, UP)
        arrow_out = Arrow(last_cell.get_right(), output_vec.get_left(), color=COLOR_LSTM)
        packet_out = Dot(color=COLOR_PACKET).move_to(last_cell.get_center())
        self.play(Create(arrow_out))
        self.play(
            packet_out.animate.move_to(output_vec.get_center()),
            GrowFromCenter(output_vec),
            Write(output_label)
        )
        self.play(FadeOut(packet_out))
        self.single_lstm_group = VGroup(
            inputs, lstm_cells, arrows_input, arrows_recurrence, 
            dropout_dots, eq_lstm, output_vec, arrow_out, output_label
        )

    def play_parallel_processing(self):
        """
        Detailed LSTM is transformed into a simplified icon and arranged into a 5x6 grid,
        representing 50 parallel LSTM computations (demonstration only).
        """
        simple_box = Rectangle(height=0.5, width=1.0, color=COLOR_LSTM, fill_opacity=0.5)
        simple_arrow = Arrow(LEFT, RIGHT, stroke_width=2).scale(0.3).next_to(simple_box, RIGHT, buff=0.1)
        simple_vec = Line(UP, DOWN, color=COLOR_EMBED).scale(0.3).next_to(simple_arrow, RIGHT, buff=0.1)
        simple_icon = VGroup(simple_box, simple_arrow, simple_vec).center()
        self.play(
            ReplacementTransform(self.single_lstm_group, simple_icon),
            run_time=1.5
        )

        rows, cols = 5, 6
        max_parallel = 30
        grid_group = VGroup()

        x_spacing = 2
        y_spacing = 1.5
        total_width = (cols - 1) * x_spacing
        total_height = (rows - 1) * y_spacing
        grid_center = ORIGIN

        start_x = grid_center[0] - total_width / 2
        start_y = grid_center[1] + total_height / 2

        for i in range(rows):
            for j in range(cols):
                if len(grid_group) >= max_parallel:
                    break
                clone = simple_icon.copy()
                # Calculate balanced position
                x = start_x + j * x_spacing
                y = start_y - i * y_spacing
                clone.move_to(np.array([x, y, 0]))
                grid_group.add(clone)
            if len(grid_group) >= max_parallel:
                break
        self.play(simple_icon.animate.move_to(grid_group[0].get_center()))
        self.play(
            LaggedStart(
                *[FadeIn(grid_group[k]) for k in range(1, max_parallel)],
                lag_ratio=0.03
            ),
            run_time=2
        )
        title_parallel = Text("Parallel Batch Processing", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title_parallel))
        self.play(
            grid_group.animate.set_color(COLOR_PACKET),
            rate_func=there_and_back,
            run_time=0.5
        )
        self.play(FadeOut(Group(*self.mobjects)))