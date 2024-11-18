from manim import *
# import random
import numpy as np

class HarmonicSeriesAndLog(Scene):
    def construct(self):
        Text.set_default(color="#1a1a1a")
        MathTex.set_default(color="#1a1a1a")

        # starting position
        r = 10
        # end pos
        n = 20

        # Create axes
        axes = Axes(
            x_range=[r, n, 1],
            y_range=[0, 4, 1],
            x_axis_config={"include_numbers": False, "color": "#1a1a1a"},
            y_axis_config={"include_numbers": True, "color": "#1a1a1a"},
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
        sum_label = MathTex(r"\sum_{i=r}^{n} \frac{1}{i}", color="#ff2e20").next_to(
            axes.c2p(n - 1, 4), RIGHT
        )

        # Transform to ln(x) label
        ln_label = MathTex(r"\ln(x)", color="#04b4d8").next_to(axes.c2p(n - 1, 4), RIGHT)

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
                width=1, height=0, fill_opacity=0.5, fill_color="#ff2e20", stroke_color="#ff2e20"
            ).move_to(axes.c2p(x_pos, 0))

            filled_rectangle = Rectangle(
                width=1,
                height=target_height,
                fill_opacity=0.5,
                fill_color="#ff2e20",
                stroke_color="#ff2e20",
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
            color="#04b4d8",
            x_range=[r, n],
        )

        # Transform the sum label to ln(x) label
        self.play(
            Create(log_graph),
            Transform(sum_label, ln_label),
        )

        # Add formula
        discrete_sum = MathTex(r"\sum_{i=r}^{n} \frac{1}{i}").scale(1.2).set_color("#ff2e20")
        integral = (
            MathTex(r"\int_{r}^{n} \frac{1}{x} \,dx\ ").scale(0.8).set_color("#04b4d8")
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
        Text.set_default(color="#1a1a1a")
        MathTex.set_default(color="#1a1a1a")
        # n_investors = 8
        investor_values = [4, 3, 3, 5, 2, 3, 8, 4]
        # Create 10 investor icons as circles
        investors = VGroup(*[Circle(radius=0.5) for _ in range(len(investor_values))])
        investors.arrange(RIGHT, buff=0.5)  # Arrange them in a row
        investors.set_fill("#434343")  # Initially gray
        investors.set_stroke(width=2)  # Set border width

        # Create labels for each investor
        labels = VGroup(*[Text(str(investor)) for investor in investor_values])
        for i, label in enumerate(labels):
            label.move_to(investors[i].get_center())

        # Show the investors and labels
        self.play(FadeIn(investors), FadeIn(labels))

        # Create the moving brace (initially placed next to the first 4 investors)
        brace = Brace(VGroup(*investors[:4]), DOWN, color="#1a1a1a")
        brace_label = brace.get_text('4 candidates being "looked" at')

        # Show the initial brace for selecting 2 investors
        self.play(FadeIn(brace), Write(brace_label))

        # Animate the selection of 2 investors
        select_4 = VGroup(investors[:4])
        select_4_labels = VGroup(labels[:4])
        brace_label.target = brace.get_text("4 candidates rejected")
        self.play(*[investor.animate.set_fill("#d62e1c") for investor in select_4], *[label.animate.set_color("#d62e1c") for label in select_4_labels], MoveToTarget(brace_label))

        # Clear the selection and text
        #self.play(*[investor.animate.set_fill(GRAY) for investor in select_2])
        #self.play(*[label.animate.set_color("#1a1a1a") for label in select_2_labels])
        #self.play(FadeOut(too_small_text))

        # Move the brace to select 5 investors
        #brace.generate_target()
        #brace.target = Brace(VGroup(*investors[:4]), DOWN, color="#1a1a1a")
        #brace_label.generate_target()
        

        # Animate the moving of the brace and changing the text
        #self.play(MoveToTarget(brace), MoveToTarget(brace_label))

        # Animate the selection of 5 investors
        brace_best = Brace(VGroup(*investors[3:4]), UP, color="#1a1a1a")
        brace_best_label = brace_best.get_text('Best candidate')
        select_4th = VGroup(investors[3:4])
        select_4th_labels = VGroup(labels[3:4])
        self.play(*[investor.animate.set_fill("#f0d53e") for investor in select_4th])
        self.play(*[label.animate.set_color("#f0d53e") for label in select_4th_labels], FadeIn(brace_best), Write(brace_best_label))

        self.wait(2)

        brace_consider = Brace(VGroup(*investors[4:8]), DOWN, color="#1a1a1a")
        brace_consider_label = brace_consider.get_text('Candidates being considered')

        select_5th = VGroup(investors[4:5])
        select_5th_labels = VGroup(labels[4:5])
        self.play(FadeOut(brace), FadeOut(brace_best), FadeOut(brace_label), FadeOut(brace_best_label), FadeIn(brace_consider), Write(brace_consider_label), *[investor.animate.set_fill("#d62e1c") for investor in select_5th], *[label.animate.set_color("#d62e1c") for label in select_5th_labels])

        select_6th = VGroup(investors[5:6])
        select_6th_labels = VGroup(labels[5:6])
        self.play(*[investor.animate.set_fill("#d62e1c") for investor in select_6th], *[label.animate.set_color("#d62e1c") for label in select_6th_labels])

        select_7th = VGroup(investors[6:7])
        select_7th_labels = VGroup(labels[6:7])
        brace_best_in_consider = Brace(VGroup(*investors[6:7]), UP, color="#1a1a1a")
        brace_best_in_consider_label = brace_best_in_consider.get_text('Better than 5')
        
        self.play(*[investor.animate.set_fill("#86CF64") for investor in select_7th], *[label.animate.set_color("#86CF64") for label in select_7th_labels], FadeIn(brace_best_in_consider), Write(brace_best_in_consider_label))


class ExampleWithTwo(Scene):
    def construct(self):
        Text.set_default(color="#1a1a1a")
        MathTex.set_default(color="#1a1a1a")
        # n_investors = 8
        student_values = [1.7, 0.3]
        # Create 10 investor icons as circles
        students = VGroup(*[Circle(radius=1) for _ in range(len(student_values))])
        students.arrange(RIGHT, buff=0.5)  # Arrange them in a row
        students.set_fill("#434343")  # Initially gray
        students.set_stroke(width=2)  # Set border width

        labels = VGroup(*[Text(str(student)) for student in student_values])
        for i, label in enumerate(labels):
            label.move_to(students[i].get_center())
        
        self.play(FadeIn(students), FadeIn(labels))

        select_1 = VGroup(students[:1])
        select_1_labels = VGroup(labels[:1])
        select_2 = VGroup(students[1:2])
        select_2_labels = VGroup(labels[1:2])
        
        self.play(*[student.animate.set_fill("#86CF64") for student in select_1])
        self.play(*[label.animate.set_color("#86CF64") for label in select_1_labels])
        self.wait(1)
        
        self.play(*[student.animate.set_fill("#d62e1c") for student in select_1], *[student.animate.set_fill("#86CF64") for student in select_2])
        self.play(*[label.animate.set_color("#d62e1c") for label in select_1_labels], *[label.animate.set_color("#86CF64") for label in select_2_labels])
        self.wait(1)

class ExampleWithThree(Scene):
    def construct(self):
        Text.set_default(color="#1a1a1a")
        MathTex.set_default(color="#1a1a1a")
        # n_investors = 8
        student_values = [0.2, -1.9, 2.4]
        # Create 10 investor icons as circles
        students = VGroup(*[Circle(radius=1) for _ in range(len(student_values))])
        students.arrange(RIGHT, buff=0.5)  # Arrange them in a row
        students.set_fill("#434343")  # Initially gray
        students.set_stroke(width=2)  # Set border width

        labels = VGroup(*[Text(str(student)) for student in student_values])
        for i, label in enumerate(labels):
            label.move_to(students[i].get_center())
        
        self.play(FadeIn(students), FadeIn(labels))

        select_1 = VGroup(students[:1])
        select_1_labels = VGroup(labels[:1])
        select_2 = VGroup(students[1:2])
        select_2_labels = VGroup(labels[1:2])
        select_3 = VGroup(students[2:3])
        select_3_labels = VGroup(labels[2:3])
        
        self.play(*[student.animate.set_fill("#86CF64") for student in select_1])
        self.play(*[label.animate.set_color("#86CF64") for label in select_1_labels])
        
        self.play(*[student.animate.set_fill("#d62e1c") for student in select_1], *[student.animate.set_fill("#86CF64") for student in select_2])
        self.play(*[label.animate.set_color("#d62e1c") for label in select_1_labels], *[label.animate.set_color("#86CF64") for label in select_2_labels])

        self.play(*[student.animate.set_fill("#d62e1c") for student in select_1], *[student.animate.set_fill("#d62e1c") for student in select_2], *[student.animate.set_fill("#86CF64") for student in select_3])
        self.play(*[label.animate.set_color("#d62e1c") for label in select_1_labels], *[label.animate.set_color("#d62e1c") for label in select_2_labels], *[label.animate.set_color("#86CF64") for label in select_3_labels])

class FinalExampleWithThree(Scene):
    def construct(self):
        Text.set_default(color="#1a1a1a")
        MathTex.set_default(color="#1a1a1a")

        student_values = [-1.9, 0.2, 2.4]

        # Create the headers
        headers = VGroup(
            Text("Reject").scale(0.8),
            Text("Compare").scale(0.8),
            Text("Result").scale(0.8)
        ).arrange(RIGHT, buff=1.5)
        headers.move_to(UP * 2)

        case1 = VGroup(
            *[Circle(radius=1) for _ in range(len(student_values))]
        ).arrange(RIGHT, buff=1.5).scale(0.7)
        case1.move_to(LEFT * 1)

        # Add headers to the scene
        self.play(FadeIn(headers))
        self.play(FadeIn(case1))
        self.wait(2)

class iMinusOne(Scene):
    def construct(self):
        Text.set_default(color="#1a1a1a")
        MathTex.set_default(color="#1a1a1a")
        # n_investors = 8
        investor_values_one = [1, 5, 3, 8, 4]
        # Create 10 investor icons as circles
        investors = VGroup(*[Circle(radius=0.5) for _ in range(len(investor_values_one))])
        investors.arrange(RIGHT, buff=0.5)  # Arrange them in a row
        investors.set_fill("#434343")  # Initially gray
        investors.set_stroke(width=2)  # Set border width
        investors.move_to(UP * 2 + LEFT * 2)

        # Create labels for each investor
        labels = VGroup(*[Text(str(investor)) for investor in investor_values_one])
        for i, label in enumerate(labels):
            label.move_to(investors[i].get_center())

        # Show the investors and labels
        self.play(FadeIn(investors), FadeIn(labels))

        brace = Brace(VGroup(*investors[:3]), DOWN, color="#1a1a1a")
        brace_label = brace.get_text('Rejection region')

        select_1st = VGroup(investors[:1])
        select_1st_labels = VGroup(labels[:1])
        select_2nd = VGroup(investors[1:2])
        select_2nd_labels = VGroup(labels[1:2])
        select_3rd = VGroup(investors[2:3])
        select_3rd_labels = VGroup(labels[2:3])
        select_4th = VGroup(investors[3:4])
        select_4th_labels = VGroup(labels[3:4])
        brace_label.target = brace.get_text("Rejection region")
        self.play(FadeIn(brace), Write(brace_label))
        self.play(
            *[investor.animate.set_fill("#d62e1c") for investor in select_1st],
            *[label.animate.set_color("#d62e1c") for label in select_1st_labels],
            *[investor.animate.set_fill("#f0d53e") for investor in select_2nd],
            *[label.animate.set_color("#f0d53e") for label in select_2nd_labels],
            *[investor.animate.set_fill("#d62e1c") for investor in select_3rd],
            *[label.animate.set_color("#d62e1c") for label in select_3rd_labels],
            *[investor.animate.set_fill("#86CF64") for investor in select_4th],
            *[label.animate.set_color("#86CF64") for label in select_4th_labels],
            MoveToTarget(brace_label))

        Text.set_default(color="#86CF64")
        good_text = Text("Good!")
        good_text.move_to(UP * 1.5 + RIGHT * 4)
        Text.set_default(color="#1a1a1a")

        self.play(Write(good_text))

        self.wait(1)

        investor_values_two = [3, 2, 5, 7, 9]

        investors = VGroup(*[Circle(radius=0.5) for _ in range(len(investor_values_two))])
        investors.arrange(RIGHT, buff=0.5)  # Arrange them in a row
        investors.set_fill("#434343")  # Initially gray
        investors.set_stroke(width=2)  # Set border width
        investors.move_to(DOWN * 1 + LEFT * 2)

        # Create labels for each investor
        labels = VGroup(*[Text(str(investor)) for investor in investor_values_two])
        for i, label in enumerate(labels):
            label.move_to(investors[i].get_center())

        # Show the investors and labels
        self.play(FadeIn(investors), FadeIn(labels))

        brace = Brace(VGroup(*investors[:3]), DOWN, color="#1a1a1a")
        brace_label = brace.get_text('Rejection region')

        select_1st = VGroup(investors[:1])
        select_1st_labels = VGroup(labels[:1])
        select_2nd = VGroup(investors[1:2])
        select_2nd_labels = VGroup(labels[1:2])
        select_3rd = VGroup(investors[2:3])
        select_3rd_labels = VGroup(labels[2:3])
        select_4th = VGroup(investors[3:4])
        select_4th_labels = VGroup(labels[3:4])
        brace_label.target = brace.get_text("Rejection region")
        self.play(FadeIn(brace), Write(brace_label))
        self.play(
            *[investor.animate.set_fill("#d62e1c") for investor in select_1st],
            *[label.animate.set_color("#d62e1c") for label in select_1st_labels],
            *[investor.animate.set_fill("#d62e1c") for investor in select_2nd],
            *[label.animate.set_color("#d62e1c") for label in select_2nd_labels],
            *[investor.animate.set_fill("#f0d53e") for investor in select_3rd],
            *[label.animate.set_color("#f0d53e") for label in select_3rd_labels],
            *[investor.animate.set_fill("#86CF64") for investor in select_4th],
            *[label.animate.set_color("#86CF64") for label in select_4th_labels],
            MoveToTarget(brace_label))

        Text.set_default(color="#d62e1c")
        bad_text = Text("Bad!")
        bad_text.move_to(DOWN * 1.5 + RIGHT * 4)

        self.play(Write(bad_text))

class DerivationOfSumFormula(Scene):
    def construct(self):
        Text.set_default(color="#1a1a1a")
        MathTex.set_default(color="#1a1a1a")
        
        naive = MathTex(r"\Pr(\text{success})")
        sum_of_i_and_best = MathTex(
            r"\sum_{i=1}^{n} \Pr(i \text{ is selected} \land i \text{ is the smartest})"
        )
        self.play(Write(naive))
        self.play(Transform(naive, sum_of_i_and_best))
        self.wait(1)
        split_sum = MathTex(
            r"\sum_{i=1}^{r} \Pr(i \text{ is selected} \land i \text{ is the smartest}) + \\",
            r"\sum_{i=r+1}^{n} \Pr(i \text{ is selected} \land i \text{ is the smartest})",
        )

        canceled_sum = MathTex(
            r"0 + \sum_{i=r+1}^{n} \Pr(i \text{ is selected} \land i \text{ is the smartest})"
        )
        sum_of_i_given_best = MathTex(
            r"\sum_{i=r+1}^{n} \Pr(i \text{ is selected} \mid i \text{ is the smartest}) * \Pr(i \text{ is the smartest})"
        )
        sum_of_i_given_best_best_taken_out = MathTex(
            r"\frac{1}{n} * \sum_{i=r+1}^{n} \Pr(i \text{ is selected} \mid i \text { is the smartest})"
        )
        sum_of_first_i_minus_one_in_first_r = MathTex(
            r"\frac{1}{n} * \sum_{i=r+1}^{n} \Pr(\text{smartest of } i-1 \text{ in first r} \mid i \text{ is the smartest}})"
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

        integral = MathTex(r"\approx \frac{r}{n} * \int_{r+1}^{n} \frac{1}{x-1} \,dx\ ")

        integrated = MathTex(r"\frac{r}{n}*(\ln({n-1})-\ln({r}))")
        log_lawed = MathTex(r"\frac{r}{n}*(\ln(\frac{n-1}{r}))")
        log_lawed_with_thing = MathTex(r"P(r)=\frac{r}{n}*(\ln(\frac{n-1}{r}))")
        self.play(Transform(naive, integral))
        self.wait(1)
        self.play(Transform(naive, integrated))
        self.wait(1)
        self.play(Transform(naive, log_lawed))
        self.wait(1)
        self.play(Transform(naive, log_lawed_with_thing))

        derivative = MathTex(r"\frac{dP(r)}{dr}=\frac{1}{n} * (\ln({n-1})-\ln({r})-1)")
        derivativeEqualZero = MathTex(r"\frac{1}{n} * (\ln({n-1})-\ln({r})-1)=0")
        derivativeEqualZeroPtOne = MathTex(r"\ln({n-1})-\ln({r})-1=0")
        derivativeEqualZeroPtTwo = MathTex(r"\ln({r})=\ln({n-1})-1")
        derivativeEqualZeroPtThree = MathTex(r"r=e^{\ln({n-1})-1}")
        derivativeEqualZeroPtFour = MathTex(r"r=\frac{e^{\ln({n-1})}}{e}")
        derivativeEqualZeroPtFive = MathTex(r"r=\frac{n-1}{e}")
        secondDerivative = MathTex(r"\frac{d^{2}P(r)}{dr^{2}}=-\frac{1}{nr}")
        secondDerivativePtOne = MathTex(r"\frac{d^{2}P(r)}{dr^{2}}|_{\frac{n-1}{e}}=-\frac{1}{n*\frac{n-1}{e}}")
        secondDerivativePtTwo = MathTex(r"=-\frac{e}{n*{(n-1)}")

        limitThingOne = MathTex(r"{n-1} \approx n")
        limitThingTwo = MathTex(r"\lim_{n \to \infty} {n-1} = \lim_{n \to \infty} {n}")
        finalAnswer = MathTex(r"\therefore r=\frac{n}{e} \approx 37 \% * n")

        self.wait(1)
        self.play(Transform(naive, derivative))
        self.wait(1)
        self.play(Transform(naive, derivativeEqualZero))
        self.wait(1)
        self.play(Transform(naive, derivativeEqualZeroPtOne))
        self.wait(1)
        self.play(Transform(naive, derivativeEqualZeroPtTwo))
        self.wait(1)
        self.play(Transform(naive, derivativeEqualZeroPtThree))
        self.wait(1)
        self.play(Transform(naive, derivativeEqualZeroPtFour))
        self.wait(1)
        self.play(Transform(naive, derivativeEqualZeroPtFive))
        self.wait(1)
        self.play(Transform(naive, secondDerivative))
        self.wait(1)
        self.play(Transform(naive, secondDerivativePtOne))
        self.wait(1)
        self.play(Transform(naive, secondDerivativePtTwo))
        self.wait(1)
        self.play(Transform(naive, limitThingOne))
        self.wait(1)
        self.play(Transform(naive, limitThingTwo))
        self.wait(1)
        self.play(Transform(naive, finalAnswer))
        self.wait(1)

        subIntoPPtOne = MathTex(r"P(r)=\frac{r}{n}*(\ln(\frac{n}{r}))")
        subIntoPPtTwo = MathTex(r"P(\frac{n}{e})=\frac{\frac{n}{e}}{n}*(\ln(\frac{n}{\frac{n}{e}}))")
        subIntoPPtThree = MathTex(r"=\frac{1}{e}*(\ln(e))")
        subIntoPPtFour = MathTex(r"P(r)=\frac{1}{e} \approx 37 \%")

        self.wait(1)
        self.play(Transform(naive, subIntoPPtOne))
        self.wait(1)
        self.play(Transform(naive, subIntoPPtTwo))
        self.wait(1)
        self.play(Transform(naive, subIntoPPtThree))
        self.wait(1)
        self.play(Transform(naive, subIntoPPtFour))

        self.wait(2)

class OptimizingForR(Scene):
    def construct(self):
        Text.set_default(color="#1a1a1a")
        MathTex.set_default(color="#1a1a1a")
        # Create the axes
        axes = Axes(
            x_range=[0, 1, 0.1],
            y_range=[0, 0.4, 0.1],
            x_axis_config={"include_numbers": True, "color": "#1a1a1a"},
            y_axis_config={"include_numbers": True, "color": "#1a1a1a"},
            tips=False
        )
        
        # Add x and y labels
        x_label = axes.get_x_axis_label(MathTex(r"x=\frac{r}{n}"))
        y_label = axes.get_y_axis_label("y")
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Plot the main function
        log_graph = axes.plot(
            lambda x: -x * np.log(x),
            color="#04b4d8",
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
                color="#ff2e20"
            )
            return line

        # Dot to represent the point of tangency
        dot = always_redraw(lambda: Dot(axes.c2p(x_start.get_value(), -x_start.get_value() * np.log(x_start.get_value())), color="#ff2e20"))
        
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
            color="#ffd966",
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
