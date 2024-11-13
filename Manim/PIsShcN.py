from manim import *
from manim import config as global_config
config = global_config.copy()

from manim import WHITE
config.background_color = WHITE

import random
import numpy as np


class HarmonicSeriesAndLog(Scene):
    def construct(self):

        # starting position
        r = 10
        # end pos
        n = 20

        # Create axes
        axes = Axes(
            x_range=[r, n, 1],
            y_range=[0, 4, 1],
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": True},
            tips=False,
        )

        # Labels
        x_labels = {i: f"r+{i-r}" if i - r > 0 else f"r={r}" for i in range(r, n + 1)}
        axes.x_axis.add_labels(x_labels)
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")

        # Create initial rectangles with height 0
        rectangles = VGroup()

        # Initial sum label
        sum_label = MathTex(r"\sum_{i=r}^{n} \frac{1}{i}", color=RED).next_to(
            axes.c2p(n - 1, 4), RIGHT
        )

        # Transform to ln(x) label
        ln_label = MathTex(r"\ln(x)", color=BLUE).next_to(axes.c2p(n - 1, 4), RIGHT)

        # Create the scene
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait()

        # Create all rectangles at once with height 0
        self.play(Create(rectangles))
        self.play(Write(sum_label))

        # Animate rectangles growing to their final heights
        # approximate it, crba to calc it
        current_sum = np.log(r)
        for i in range(r, n):
            current_sum += 1 / (i + 1)
            target_height = current_sum * axes.y_axis.get_unit_size()
            x_pos = i + 0.5

            empty_rectangle = Rectangle(
                width=1, height=0, fill_opacity=0.5, fill_color=RED, stroke_color=RED
            ).move_to(axes.c2p(x_pos, 0))

            filled_rectangle = Rectangle(
                width=1,
                height=target_height,
                fill_opacity=0.5,
                fill_color=RED,
                stroke_color=RED,
            ).move_to(axes.c2p(x_pos, current_sum / 2))

            dot = Dot(
                axes.c2p(x_pos + 0.5 - DEFAULT_DOT_RADIUS, current_sum), color=PURPLE
            )

            # self.play(Create(empty_rectangle))
            self.play(
                Transform(empty_rectangle, filled_rectangle),
                FadeIn(dot),
                run_time=0.5,
            )
            # self.play(FadeIn(dot))

        self.wait()

        # Now add the natural log function
        log_graph = axes.plot(
            lambda x: np.log(x),
            color=BLUE,
            x_range=[r, n],
        )

        # Transform the sum label to ln(x) label
        self.play(
            Create(log_graph),
            Transform(sum_label, ln_label),
        )

        # Add formula
        discrete_sum = MathTex(r"\sum_{i=r}^{n} \frac{1}{i}").scale(1.2).set_color(RED)
        integral = (
            MathTex(r"\int_{r}^{n} \frac{1}{x} \,dx\ ").scale(0.8).set_color(BLUE)
        )

        discrete_sum.to_edge(UP)
        integral.to_edge(UP)

        # Write discrete sum first
        self.play(Write(discrete_sum))

        # Animate the transformation from the discrete sum to the integral
        self.play(
            Transform(discrete_sum, integral),  # Transform the sum into the integral
            run_time=1,  # You can adjust the run time for the transformation speed
        )

        self.wait(2)  # Wait for 2 seconds to show the result


class SamplingSceneWithBrace(Scene):
    def construct(self):
        # n_investors = 8
        investor_values = [4, 3, 3, 5, 8, 2, 3, 4]
        # Create 10 investor icons as circles
        investors = VGroup(*[Circle(radius=0.5) for _ in range(len(investor_values))])
        investors.arrange(RIGHT, buff=0.5)  # Arrange them in a row
        investors.set_fill(GRAY)  # Initially gray
        investors.set_stroke(width=2)  # Set border width

        # Create labels for each investor
        labels = VGroup(*[Text(str(investor)) for investor in investor_values])
        for i, label in enumerate(labels):
            label.move_to(investors[i].get_center())

        # Show the investors and labels
        self.play(FadeIn(investors), FadeIn(labels))

        # Create the moving brace (initially placed next to the first 2 investors)
        brace = Brace(VGroup(investors[0], investors[1]), DOWN)
        brace_label = brace.get_text("2 investors selected")

        # Show the initial brace for selecting 2 investors
        self.play(FadeIn(brace), Write(brace_label))

        # Animate the selection of 2 investors
        select_2 = VGroup(investors[0], investors[1])
        select_2_labels = VGroup(labels[0], labels[1])
        self.play(*[investor.animate.set_fill(GREEN) for investor in select_2])
        self.play(*[label.animate.set_color(GREEN) for label in select_2_labels])

        # Comment on 2 being too small
        too_small_text = Tex("Too few samples?").shift(UP * 2)
        self.play(Write(too_small_text))
        # self.wait(1)

        # Clear the selection and text
        self.play(*[investor.animate.set_fill(GRAY) for investor in select_2])
        self.play(*[label.animate.set_color(WHITE) for label in select_2_labels])
        self.play(FadeOut(too_small_text))

        # Move the brace to select 5 investors
        brace.generate_target()
        brace.target = Brace(VGroup(*investors[:5]), DOWN)
        brace_label.generate_target()
        brace_label.target = brace.get_text("5 investors selected")

        # Animate the moving of the brace and changing the text
        self.play(MoveToTarget(brace), MoveToTarget(brace_label))

        # Animate the selection of 5 investors
        select_5 = VGroup(investors[:5])
        select_5_labels = VGroup(labels[:5])
        self.play(*[investor.animate.set_fill(GREEN) for investor in select_5])
        self.play(*[label.animate.set_color(GREEN) for label in select_5_labels])

        # Comment on 5 being too large
        too_large_text = Tex("Too many samples?").shift(UP * 2)
        self.play(Write(too_large_text))
        self.wait(1)

        # Clear the selection and text
        self.play(*[investor.animate.set_fill(GRAY) for investor in select_5])
        self.play(*[label.animate.set_color(WHITE) for label in select_5_labels])
        self.play(FadeOut(too_large_text), FadeOut(brace), FadeOut(brace_label))

        # End the scene with a final comment or conclusion
        conclusion_text = Tex("Finding the right number of samples is key.").shift(
            UP * 3
        )
        self.play(Write(conclusion_text))
        self.play(FadeOut(conclusion_text), FadeOut(investors), FadeOut(labels))

        self.wait(2)


class DerivationOfSumFormula(Scene):
    def construct(self):
        naive = MathTex(r"\Pr(\text{success})")
        sum_of_i_and_best = MathTex(
            r"\sum_{i=1}^{n} \Pr(i \text{ is selected} \land i \text{ is the best})"
        )
        self.play(Write(naive))
        self.play(Transform(naive, sum_of_i_and_best))
        self.wait(1)
        split_sum = MathTex(
            r"\sum_{i=1}^{r} \Pr(i \text{ is selected} \land i \text{ is the best}) + \\",
            r"\sum_{i=r+1}^{n} \Pr(i \text{ is selected} \land i \text{ is the best})",
        )

        canceled_sum = MathTex(
            r"0 + \sum_{i=r+1}^{n} \Pr(i \text{ is selected} \land i \text{ is the best})"
        )
        sum_of_i_given_best = MathTex(
            r"\sum_{i=r+1}^{n} \Pr(i \text{ is selected} \mid i \text{ is the best}) * \Pr(i \text{ is the best})"
        )
        sum_of_i_given_best_best_taken_out = MathTex(
            r"\frac{1}{n} * \sum_{i=r+1}^{n} \Pr(i \text{ is selected} \mid i \text { is the best})"
        )
        sum_of_first_i_minus_one_in_first_r = MathTex(
            r"\frac{1}{n} * \sum_{i=r+1}^{n} \Pr(\text{best of } i-1 \text{ in first r} \mid i \text{ is the best}})"
        )
        sum_of_first_i_minus_one_in_first_r_maths = MathTex(
            r"\frac{1}{n} * \sum_{i=r+1}^{n} \frac{r}{i-1}"
        )
        sum_of_first_i_minus_one_in_first_r_maths_r_taken_out = MathTex(
            r"\frac{r}{n} * \sum_{i=r+1}^{n} \frac{1}{i-1}"
        )
        self.play(Transform(naive, split_sum))
        self.wait(0.5)
        self.play(Transform(naive, canceled_sum))
        self.wait(1)
        self.play(Transform(naive, sum_of_i_given_best))
        self.wait(1)
        self.play(Transform(naive, sum_of_i_given_best_best_taken_out))
        self.wait(1)
        self.play(Transform(naive, sum_of_first_i_minus_one_in_first_r))
        self.wait(1)
        self.play(Transform(naive, sum_of_first_i_minus_one_in_first_r_maths))
        self.wait(1)
        self.play(
            Transform(naive, sum_of_first_i_minus_one_in_first_r_maths_r_taken_out)
        )
        self.wait(2)

        # show the integral approximation thing there

        integral = MathTex(r"\approx \frac{r}{n} * \int_{r}^{n} \frac{1}{x-1} \,dx\ ")

        integrated = MathTex(r"\frac{r}{n}*(\ln({n-1})-\ln({r-1}))")
        log_lawed = MathTex(r"\frac{r}{n}*(\ln(\frac{n-1}{r-1}))")
        curious_approximation_of_fraction = MathTex(r"\approx \frac{r}{n}*(\ln(\frac{n}{r}))")
        self.play(Transform(naive, integral))
        self.wait(1)
        self.play(Transform(naive, integrated))
        self.wait(1)
        self.play(Transform(naive, log_lawed))
        self.wait(1)
        self.play(Transform(naive, curious_approximation_of_fraction))
        
        self.play(FadeOut(naive))

        self.wait(2)

class OptimizingForR(Scene):
    def construct(self):
        # Create the axes
        axes = Axes(
            x_range=[0, 1, 0.1],
            y_range=[0, 0.4, 0.1],
            x_axis_config={"include_numbers": True},
            y_axis_config={"include_numbers": True},
            tips=False,
        )
        
        # Add x and y labels
        x_label = axes.get_x_axis_label(MathTex(r"x=\frac{r}{n}"))
        y_label = axes.get_y_axis_label("y")
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Plot the main function
        log_graph = axes.plot(
            lambda x: -x * np.log(x),
            color=BLUE,
            x_range=[0.001, 1],
        )
        self.play(Create(log_graph))
        
        # Define the derivative of the function: f'(x) = -ln(x) - 1
        def derivative(x):
            return -np.log(x) - 1
        
        # Define initial point for the tangent line
        x_start = ValueTracker(0.001)
        
        # Function to create tangent line
        def get_tangent_line():
            x = x_start.get_value()
            y = -x * np.log(x)
            slope = derivative(x)
            
            # Create points for the line segment
            x_range = 0.2  # Length of tangent line
            x1 = max(0.001, x - x_range)
            x2 = min(1, x + x_range)
            
            y1 = y + slope * (x1 - x)
            y2 = y + slope * (x2 - x)
            
            line = Line(
                start=axes.c2p(x1, y1),
                end=axes.c2p(x2, y2),
                color=RED
            )
            return line

        # Dot to represent the point of tangency
        dot = always_redraw(lambda: Dot(axes.c2p(x_start.get_value(), -x_start.get_value() * np.log(x_start.get_value())), color=RED))
        
        # Create the tangent line that updates with the dot
        tangent_line = always_redraw(get_tangent_line)
        
        self.play(FadeIn(dot), Create(tangent_line))

        # Move the dot and watch the tangent line follow
        self.play(
            x_start.animate.set_value(1/np.e),
            run_time=3
        )

        self.wait()

        # Create dashed line at x = 1/e
        final_x = 1/np.e
        final_y = -final_x * np.log(final_x)
        dashed_line = DashedLine(
            start=axes.c2p(final_x, final_y),
            end=axes.c2p(final_x, 0),
            color=YELLOW,
            dash_length=0.1
        )

        # Add label for 1/e
        one_over_e_label = MathTex(r"\frac{1}{e}").scale(0.5)
        one_over_e_label.move_to(axes.c2p(final_x+0.01, 0.02))

        self.play(
            Create(dashed_line),
            Write(one_over_e_label)
        )

        self.wait(2)
