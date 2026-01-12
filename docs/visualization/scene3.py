from manim import *
import numpy as np

COLOR_LSTM = "#9370DB"
COLOR_EMBED = "#FFA500"
COLOR_COMBINED = "#90EE90"
COLOR_NORM = "#3CB371"
COLOR_TEXT = WHITE
COLOR_GRID = GRAY

class Scene3_EmbeddingConcatenation(Scene):
    def construct(self):
        embed_vec = self.play_embedding_lookup()
        combined_vec = self.play_vector_fusion(embed_vec)
        norm_vec, full_group = self.play_layernorm(combined_vec)
        self.play_transition(norm_vec, full_group)

    def setup_background(self):
        grid = NumberPlane(
            background_line_style={
                "stroke_color": COLOR_GRID,
                "stroke_width": 1,
                "stroke_opacity": 0.2
            }
        )
        self.add(grid)

        labels = VGroup(
            Text("AAPL: ID=0", font_size=18, color=GRAY).to_corner(UL),
            Text("MSFT: ID=1", font_size=18, color=GRAY).to_corner(UR),
            Text("NVDA: ID=2", font_size=18, color=GRAY).to_corner(DL),
        )
        self.add(labels)

    def play_embedding_lookup(self):
        stock_id = MathTex(r"\text{Stock\_ID} = 0", font_size=36).move_to(LEFT * 5 + UP * 2)
        matrix_group = VGroup()
        rows, cols = 5, 4
        for r in range(rows):
            row_group = VGroup()
            for c in range(cols):
                val = f"{np.random.uniform(-1, 1):.2f}"
                txt = Text(val, font_size=16, color=GRAY_B)
                box = Rectangle(height=0.5, width=1.0, stroke_color=GRAY, stroke_width=1)
                cell = VGroup(box, txt)
                row_group.add(cell)
            row_group.arrange(RIGHT, buff=0)
            matrix_group.add(row_group)
        matrix_group.arrange(DOWN, buff=0).move_to(LEFT * 2)
        matrix_header = Text("Embedding Matrix [53, 16]", font_size=24, color=COLOR_EMBED).next_to(matrix_group, UP)

        self.play(
            FadeIn(stock_id),
            FadeIn(matrix_group),
            FadeIn(matrix_header)
        )

        target_row = matrix_group[0]
        arrow = Arrow(stock_id.get_right(), target_row.get_left(), color=COLOR_EMBED)
        self.play(GrowArrow(arrow))

        self.play(
            target_row.animate.set_color(COLOR_EMBED).scale(1.05),
            Indicate(target_row, color=YELLOW)
        )

        embed_vec_rect = Rectangle(width=1.5, height=0.6, fill_color=COLOR_EMBED, fill_opacity=0.7, stroke_width=2)
        embed_lbl = MathTex(r"[\text{Embed}=16]", font_size=20, color=WHITE).move_to(embed_vec_rect)
        embed_vec = VGroup(embed_vec_rect, embed_lbl).move_to(DOWN * 2 + RIGHT * 2)

        self.play(
            ReplacementTransform(target_row.copy(), embed_vec),
            FadeOut(arrow)
        )

        self.matrix_stuff = VGroup(stock_id, matrix_group, matrix_header)
        return embed_vec

    def play_vector_fusion(self, embed_vec):
        lstm_rect = Rectangle(width=4, height=0.6, fill_color=COLOR_LSTM, fill_opacity=0.7, stroke_width=2)
        lstm_lbl = MathTex(r"[\text{LSTM Hidden}=128]", font_size=20, color=WHITE).move_to(lstm_rect)
        lstm_vec = VGroup(lstm_rect, lstm_lbl).move_to(UP * 2 + RIGHT * 2)
        self.play(FadeIn(lstm_vec))
        self.play(
            embed_vec.animate.next_to(lstm_vec, RIGHT, buff=0.1)
        )

        combined_rect = Rectangle(width=5.5, height=0.6, fill_color=COLOR_COMBINED, fill_opacity=0.7, stroke_width=2)
        combined_lbl = MathTex(r"[\text{Combined}=144]", font_size=20, color=WHITE).move_to(combined_rect)
        combined_vec = VGroup(combined_rect, combined_lbl).move_to(RIGHT * 4)
        center_target = VGroup(lstm_vec, embed_vec).get_center()
        combined_vec.move_to(center_target)
        self.play(
            ReplacementTransform(VGroup(lstm_vec, embed_vec), combined_vec)
        )

        fusion_text = Text("Fusion: Temporal Dynamics + Stock Traits", font_size=24, color=COLOR_COMBINED).next_to(combined_vec, UP)
        self.play(
            Indicate(combined_vec, color=WHITE, scale_factor=1.05),
            FadeIn(fusion_text)
        )

        return combined_vec

    def play_layernorm(self, combined_vec):
        eq_tex = r"x' = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta"
        layernorm_eq = MathTex(eq_tex, color=COLOR_COMBINED).scale(0.9).move_to(DOWN * 2.5)
        self.play(
            Write(layernorm_eq),
            FadeOut(self.matrix_stuff)
        )

        norm_arrow = Arrow(combined_vec.get_bottom(), layernorm_eq.get_top(), color=COLOR_COMBINED, buff=0.2)
        self.play(GrowArrow(norm_arrow))

        sample_vals = VGroup(*[MathTex(v, font_size=24) for v in ["1.2", "-0.5", "0.8", "...", "2.1"]])
        sample_vals.arrange(RIGHT, buff=0.3).next_to(combined_vec, DOWN, buff=0.5)
        self.play(FadeIn(sample_vals))

        stats_group = VGroup(
            MathTex(r"-\mu", color=RED).scale(0.75),
            MathTex(r"/\sigma", color=RED).scale(0.75)
        ).arrange(RIGHT, buff=0.1).next_to(sample_vals, RIGHT)
        self.play(
            FadeIn(stats_group),
            sample_vals.animate.set_color(YELLOW)
        )

        self.play(
            Indicate(layernorm_eq.get_part_by_tex(r"\gamma"), color=YELLOW, scale_factor=1.5),
            Indicate(layernorm_eq.get_part_by_tex(r"\beta"), color=YELLOW, scale_factor=1.5)
        )

        norm_rect = Rectangle(width=5.5, height=0.6, fill_color=COLOR_NORM, fill_opacity=0.8, stroke_width=2)
        norm_lbl = MathTex(r"[\text{Normalized}=144]", font_size=20, color=WHITE).move_to(norm_rect)
        norm_vec = VGroup(norm_rect, norm_lbl).move_to(combined_vec.get_center())

        stab_text = Text("Stabilizes for Transformer", font_size=24, color=COLOR_NORM).move_to(RIGHT * 3 + UP * 2)

        self.play(
            ReplacementTransform(combined_vec, norm_vec),
            FadeOut(sample_vals),
            FadeOut(stats_group),
            FadeIn(stab_text)
        )

        full_group = VGroup(norm_vec, layernorm_eq, norm_arrow, stab_text)
        return norm_vec, full_group

    def play_transition(self, norm_vec, full_group):
        self.play(
            full_group.animate.scale(0.7).to_edge(LEFT, buff=1)
        )

        nodes = VGroup(*[Circle(radius=0.2, color=COLOR_COMBINED, fill_opacity=0.3) for _ in range(4)])
        nodes.arrange_in_grid(rows=2, cols=2, buff=0.5).move_to(RIGHT * 4)
        lines = VGroup()
        for n1 in nodes:
            for n2 in nodes:
                if n1 != n2:
                    lines.add(Line(n1.get_center(), n2.get_center(), stroke_width=0.5, stroke_opacity=0.3, color=COLOR_COMBINED))
        teaser = VGroup(lines, nodes)
        self.play(FadeIn(teaser))

        recap_text = Text("Ready for Cross-Sectional Attention", font_size=28, color=WHITE).next_to(teaser, UP)
        self.play(
            Write(recap_text),
            Indicate(norm_vec, color=WHITE)
        )