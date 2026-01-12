from manim import *
from scipy.stats import norm
import numpy as np

COLOR_LSTM = "#9370DB"
COLOR_EMBED = "#FFA500"
COLOR_FUSION = "#90EE90"
COLOR_ATTN = "#00BFFF"
COLOR_MU = BLUE
COLOR_VAR = RED
COLOR_TARGET = GREEN
COLOR_RANK = ORANGE
COLOR_LOSS = RED

class Scene3_HybridArchitecture(ThreeDScene):
    def construct(self):
        self.play_input_cube()
        self.play_lstm_path()
        self.play_identity_path()
        self.play_fusion()
        self.play_transformer_network()
        self.play_probabilistic_head()
        self.play_ranking_head()
        self.play_hybrid_interaction()

    def play_input_cube(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        depth_stock = 4.0
        width_time = 4.0
        height_feat = 2.0

        bulk_tensor = Prism(
            dimensions=[width_time, height_feat, depth_stock - 0.2],
            fill_color=BLUE,
            fill_opacity=0.2,
            stroke_color=BLUE_E,
            stroke_width=1
        ).move_to(IN * 0.2)

        slice_bg = Prism(
            dimensions=[width_time, height_feat, 0.2],
            fill_color=BLUE_E,
            fill_opacity=0.5,
            stroke_color=WHITE,
            stroke_width=2
        ).move_to(OUT * (depth_stock/2))

        grid_lines = VGroup()
        for x in np.linspace(-width_time/2, width_time/2, 10):
            l = Line(
                start=[x, -height_feat/2, depth_stock/2 + 0.1], 
                end=[x, height_feat/2, depth_stock/2 + 0.1],
                color=WHITE, stroke_width=1, stroke_opacity=0.5
            )
            grid_lines.add(l)
        for y in np.linspace(-height_feat/2, height_feat/2, 4):
            l = Line(
                start=[-width_time/2, y, depth_stock/2 + 0.1], 
                end=[width_time/2, y, depth_stock/2 + 0.1],
                color=WHITE, stroke_width=1, stroke_opacity=0.5
            )
            grid_lines.add(l)
        
        self.hero_group = VGroup(slice_bg, grid_lines)
        tensor_group = VGroup(bulk_tensor, self.hero_group)
        tensor_group.scale(0.7).shift(DOWN * 0.5)

        lbl_stocks = Text("N=53 Stocks", font_size=24).rotate(PI/2, axis=RIGHT).next_to(bulk_tensor, UP)
        lbl_time = Text("T=60 Days", font_size=24).rotate(PI/2, axis=RIGHT).next_to(self.hero_group, DOWN)
        lbl_feats = Text("F=6 Features", font_size=24).rotate(PI/2, axis=RIGHT).next_to(self.hero_group, RIGHT)
        labels_3d = VGroup(lbl_stocks, lbl_time, lbl_feats)

        self.play(DrawBorderThenFill(tensor_group), Write(labels_3d))
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(2.5)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES, run_time=1)
        self.play(
            self.hero_group.animate.shift(OUT * 0.5 + LEFT * 1.5),
            bulk_tensor.animate.set_opacity(0.05),
            FadeOut(labels_3d),
            run_time=2
        )
        slice_label = Text("Single Stock History", font_size=24, color=YELLOW).rotate(PI/2, axis=RIGHT).next_to(self.hero_group, UP)
        self.play(Write(slice_label), grid_lines.animate.set_color(YELLOW))
        self.wait(0.5)
        self.play(
            FadeOut(bulk_tensor),
            slice_label.animate.rotate(-90*DEGREES, axis=RIGHT).move_to(LEFT * 4 + UP * 1).scale(0.8),            
            FadeOut(slice_label),
            self.hero_group.animate.rotate(-90*DEGREES, axis=RIGHT).move_to(LEFT * 4 + UP * 1).scale(0.8),
            run_time=2
        )
        self.move_camera(phi=0, theta=-90*DEGREES, run_time=1.5)

    def play_lstm_path(self):
        import random

        num_visible_steps = 6
        inputs = VGroup()
        for i in range(num_visible_steps):
            dot = Circle(radius=0.25, color=BLUE, fill_opacity=0.5, stroke_width=2)
            label = MathTex(f"x_{{{i+1}}}", font_size=24).move_to(dot)
            group = VGroup(dot, label)
            inputs.add(group)
        inputs.arrange(RIGHT, buff=1.2).shift(DOWN * 2)
        self.play(
            LaggedStart(*[GrowFromPoint(inp, self.hero_group.get_center()) for inp in inputs], lag_ratio=0.1),
            run_time=1.5
        )
        self.play(FadeOut(self.hero_group))

        lstm_cells = VGroup()
        arrows_recurrence = VGroup()
        arrows_input = VGroup()
        cell_height = 1.2
        for i in range(num_visible_steps):
            cell = RoundedRectangle(corner_radius=0.2, height=cell_height, width=1.2, color=COLOR_LSTM, fill_opacity=0.2)
            cell.move_to(inputs[i].get_center() + UP * 2)
            lstm_cells.add(cell)
            arr_in = Arrow(inputs[i].get_top(), cell.get_bottom(), buff=0.1, color=BLUE, max_tip_length_to_length_ratio=0.15)
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
            packet_in = Dot(radius=0.08, color=YELLOW).move_to(inputs[i].get_center())
            anims.append(packet_in.animate.move_to(lstm_cells[i].get_center()))
            if i > 0:
                anims.append(Create(arrows_recurrence[i-1]))
                packet_rec = Dot(radius=0.08, color=YELLOW).move_to(lstm_cells[i-1].get_center())
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
        packet_out = Dot(color=YELLOW).move_to(last_cell.get_center())
        self.play(Create(arrow_out))
        self.play(
            packet_out.animate.move_to(output_vec.get_center()),
            GrowFromCenter(output_vec),
            Write(output_label)
        )
        self.play(FadeOut(packet_out))
        self.lstm_vec = output_vec[0]
        self.vec_label_t = output_vec[1]
        self.lbl_momentum = output_label

        # Fade out all LSTM intermediates so only output vector remains
        cleanup_group = VGroup(
            inputs, lstm_cells, arrows_input, arrows_recurrence, 
            dropout_dots, eq_lstm, arrow_out
        )
        self.play(FadeOut(cleanup_group))

    def play_identity_path(self):
        COLOR_PACKET = "#FFFF00"
        start_pos = LEFT * 3.5 + DOWN * 2.0

        chip_group = VGroup(
            Circle(radius=0.3, color=COLOR_EMBED, fill_opacity=0.2, stroke_width=2),
            Text("ID", font_size=16, color=COLOR_EMBED)
        ).move_to(start_pos)
        chip_label = Text("Stock Identity", font_size=25, color=COLOR_EMBED).next_to(chip_group, UP)
        self.play(FadeIn(chip_group), Write(chip_label))

        rows = 4
        matrix_group = VGroup()
        for i in range(rows):
            rect = Rectangle(height=0.25, width=1.5, color=BLUE_E, fill_opacity=0.5, stroke_width=1)
            matrix_group.add(rect)
        matrix_group.arrange(DOWN, buff=0).next_to(chip_group, RIGHT, buff=1.5)

        matrix_label = Text("Embedding Layer", font_size=16, color=BLUE).next_to(matrix_group, UP)
        arrow_in = Arrow(chip_group.get_right(), matrix_group.get_left(), color=COLOR_EMBED, buff=0.1)
        self.play(Create(matrix_group), FadeIn(matrix_label), GrowArrow(arrow_in))

        packet = Dot(radius=0.06, color=COLOR_PACKET).move_to(chip_group.get_center())
        self.play(MoveAlongPath(packet, Line(chip_group.get_center(), matrix_group.get_center())), run_time=0.6)

        target_row = matrix_group[1]
        self.play(
            FadeOut(packet),
            Indicate(target_row, color=COLOR_EMBED, scale_factor=1.2),
            run_time=0.5
        )

        self.id_vec = RoundedRectangle(height=0.4, width=1.2, corner_radius=0.1, fill_color=COLOR_EMBED, fill_opacity=0.9, stroke_width=0)
        self.id_vec.next_to(matrix_group, RIGHT, buff=1.0)
        self.vec_label_id = Text("Emb Vector", font_size=14, color=BLACK).move_to(self.id_vec)
        arrow_out = Arrow(matrix_group.get_right(), self.id_vec.get_left(), color=COLOR_EMBED)

        self.play(GrowArrow(arrow_out))
        self.play(
            TransformFromCopy(target_row, self.id_vec),
            Write(self.vec_label_id)
        )

        self.lbl_static = Text("Static Traits (Sector)", font_size=25, color=WHITE).next_to(self.id_vec, DOWN)
        self.play(FadeIn(self.lbl_static))

        self.play(
            FadeOut(chip_group), FadeOut(chip_label),
            FadeOut(matrix_group), FadeOut(matrix_label),
            FadeOut(arrow_in), FadeOut(arrow_out)
        )

    def play_fusion(self):
        center_point = ORIGIN

        self.play(
            self.lstm_vec.animate.next_to(center_point, LEFT, buff=0.1),
            self.vec_label_t.animate.next_to(center_point, LEFT, buff=0.1).shift(RIGHT*0.1),
            FadeOut(self.lbl_momentum),
            self.id_vec.animate.next_to(center_point, RIGHT, buff=0.1),
            self.vec_label_id.animate.next_to(center_point, RIGHT, buff=0.1).shift(LEFT*0.05),
            FadeOut(self.lbl_static)
        )

        brace = Brace(VGroup(self.lstm_vec, self.id_vec), UP, color=WHITE)
        concat_text = brace.get_text("Concatenate").scale(0.8)
        self.play(GrowFromCenter(brace), FadeIn(concat_text))
        self.wait(0.5)

        self.fused_vec = RoundedRectangle(height=0.5, width=4, corner_radius=0.1, fill_color=COLOR_FUSION, fill_opacity=1, stroke_width=0)
        self.fused_vec.move_to(center_point)
        self.fused_label = Text("Fused State (144 dims)", font_size=24, color=BLACK).move_to(self.fused_vec)
        self.play(
            ReplacementTransform(VGroup(self.lstm_vec, self.id_vec), self.fused_vec),
            ReplacementTransform(VGroup(self.vec_label_t, self.vec_label_id), self.fused_label),
            FadeOut(brace), FadeOut(concat_text)
        )
        self.play(Indicate(self.fused_vec, color=WHITE, scale_factor=1.05))
        self.wait(0.5)

    def play_transformer_network(self):
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, run_time=1.5)
        num_nodes = 64
        radius = 3.0
        nodes = VGroup()
        golden_ratio = (1 + 5 ** 0.5) / 2
        node_positions = []
        for i in range(num_nodes):
            theta = 2 * PI * i / golden_ratio
            phi = np.arccos(1 - 2 * (i + 0.5) / num_nodes)
            x = np.cos(theta) * np.sin(phi) * radius
            y = np.sin(theta) * np.sin(phi) * radius
            z = np.cos(phi) * radius
            pos = np.array([x, y, z])
            node_positions.append(pos)
            dot = Dot3D(point=pos, radius=0.08, color=COLOR_FUSION)
            nodes.add(dot)
        self.begin_ambient_camera_rotation(rate=0.3)
        self.play(
            ReplacementTransform(VGroup(self.fused_vec, self.fused_label), nodes),
            run_time=2
        )
        lines = VGroup()
        for i in range(num_nodes):
            targets = np.random.choice(range(num_nodes), 15, replace=False)
            for t in targets:
                if i != t:
                    dist = np.linalg.norm(node_positions[i] - node_positions[t])
                    opacity = max(0.05, 0.4 - (dist * 0.05))
                    l = Line(
                        node_positions[i], 
                        node_positions[t], 
                        stroke_width=1.5, 
                        color=COLOR_ATTN, 
                        stroke_opacity=opacity
                    )
                    lines.add(l)
        self.play(Create(lines, lag_ratio=0.001), run_time=3)
        self.play(
            nodes.animate.set_color(WHITE).scale(1.2),
            lines.animate.set_stroke(color=WHITE, opacity=0.5),
            rate_func=there_and_back,
            run_time=1.5
        )
        title = Text("Cross-Sectional Attention", font_size=32, color=COLOR_ATTN).to_corner(UL)
        sub = Text("Global Market Context (All-to-All)", font_size=24, color=GRAY).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(title, sub)
        self.play(Write(title), FadeIn(sub))
        self.wait(4)
        self.stop_ambient_camera_rotation()