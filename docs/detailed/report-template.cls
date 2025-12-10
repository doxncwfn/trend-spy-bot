\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{report-template}[22/11/25 v1.0 Report Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % Move all options to 'article' class
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{article}

\RequirePackage[utf8]{vntex,inputenc}
\RequirePackage[T5]{fontenc}
\RequirePackage{anyfontsize,amsmath,amssymb}
\RequirePackage{geometry,tikz,graphicx,indentfirst,setspace}
\RequirePackage{booktabs}
\RequirePackage[center]{caption}
\RequirePackage{fancyhdr,hyperref}
\RequirePackage[super,square,sort]{natbib}

\RequirePackage[english]{babel} % Change to '[vietnamese]' to use Vietnamese labels

% Return current date
\newcommand{\Month}{
  \ifcase \the\month
  \or January
  \or February
  \or March
  \or April
  \or May
  \or June
  \or July
  \or August
  \or September
  \or October
  \or November
  \or December
  \fi
}
\newcommand{\DateToday}{
  \the\day\ \Month
}

% Custom commands
\newcommand{\@upperuniname}{}
\newcommand{\@uniname}{TREND SPY BOT}
\newcommand{\@deptname}{REALTIME STOCK PREDICTOR SYSTEM}
\newcommand{\@coursename}{}
\newcommand{\@reporttype}{}
\renewcommand{\@title}{}
\newcommand{\@advisor}{}
\newcommand{\@stuname}{& Cuong Doan Phuong Hung & ID 2310381 \\
  & Huy Nguyen Khac Khanh & ID 2311186}
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}
\newcommand{\@reportplace}{Ho Chi Minh City}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}
\newcommand{\@reporttime}{\Month \the\year}
\newcommand{\reporttime}[1]{\renewcommand{\@reporttime}{#1}}

% Configurations
\geometry{
  paper=a4paper,
  left=2cm,
  right=2cm,
  top=1.5cm,
  bottom=1.5cm,
  includeheadfoot=true,
  headheight=30pt
}
\onehalfspacing
\raggedbottom

% Spacing settings
\renewcommand{\arraystretch}{1.2}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0.3pt}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

\AtBeginDocument{\counterwithin{equation}{section}}
\AtBeginDocument{\counterwithin{table}{section}}
\AtBeginDocument{\counterwithin{figure}{section}}

\AtBeginDocument{\renewcommand*\bibname{References}}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=red,
  urlcolor=blue
}

%Make cover page
\newcommand{\coverpage}{
  \begin{tikzpicture}[remember picture,overlay,inner sep=0,outer sep=0]
    \draw[blue!70!black,line width=4pt]
    ([xshift=-1.5cm,yshift=-2cm]current page.north east) coordinate (A)--
    ([xshift=1.5cm,yshift=-2cm]current page.north west) coordinate(B)--
    ([xshift=1.5cm,yshift=2cm]current page.south west) coordinate (C)--
    ([xshift=-1.5cm,yshift=2cm]current page.south east) coordinate(D)--cycle;

    \draw
    ([yshift=0.5cm,xshift=-0.5cm]A)--
    ([yshift=0.5cm,xshift=0.5cm]B)--
    ([yshift=-0.5cm,xshift=0.5cm]B) --
    ([yshift=-0.5cm,xshift=-0.5cm]B)--
    ([yshift=0.5cm,xshift=-0.5cm]C)--
    ([yshift=0.5cm,xshift=0.5cm]C)--
    ([yshift=-0.5cm,xshift=0.5cm]C)--
    ([yshift=-0.5cm,xshift=-0.5cm]D)--
    ([yshift=0.5cm,xshift=-0.5cm]D)--
    ([yshift=0.5cm,xshift=0.5cm]D)--
    ([yshift=-0.5cm,xshift=0.5cm]A)--
    ([yshift=-0.5cm,xshift=-0.5cm]A)--
    ([yshift=0.5cm,xshift=-0.5cm]A);

    \draw
    ([yshift=-0.3cm,xshift=0.3cm]A)--
    ([yshift=-0.3cm,xshift=-0.3cm]B)--
    ([yshift=0.3cm,xshift=-0.3cm]B) --
    ([yshift=0.3cm,xshift=0.3cm]B)--
    ([yshift=-0.3cm,xshift=0.3cm]C)--
    ([yshift=-0.3cm,xshift=-0.3cm]C)--
    ([yshift=0.3cm,xshift=-0.3cm]C)--
    ([yshift=0.3cm,xshift=0.3cm]D)--
    ([yshift=-0.3cm,xshift=0.3cm]D)--
    ([yshift=-0.3cm,xshift=-0.3cm]D)--
    ([yshift=0.3cm,xshift=-0.3cm]A)--
    ([yshift=0.3cm,xshift=0.3cm]A)--
    ([yshift=-0.3cm,xshift=0.3cm]A);
  \end{tikzpicture}

  \pagestyle{empty}
  \vspace*{-3\baselineskip}

  \begin{center}
    \MakeUppercase{\@upperuniname}

    \MakeUppercase{\@uniname}

    \MakeUppercase{\@deptname}
    \vfill
    \includegraphics[scale=0.45]{stock-icon.png}
    \vfill
    \bfseries
    \begin{tabular}{c}
      \Large\@coursename                                 \\
      {}                                                 \\
      \midrule                                           \\
      \Large\@reporttype                                 \\
      {}                                                 \\
      \Huge\parbox[c]{0.8\linewidth}{\centering \@title} \\
      {}                                                 \\
      \midrule
    \end{tabular}
    \vfill
    \mdseries
    \begin{tabular}{rll}
      Advisor: \@advisor  \\
      Students: \@stuname \\
    \end{tabular}
    \vfill

    \MakeUppercase{\@reportplace, \@reporttime}
  \end{center}

  \clearpage
  \pagestyle{fancy}
  \newpage
}

% Header and footer
\pagestyle{fancy}
\fancyhf{}

\if@twoside
  \fancyhead[EL]{
    \bfseries
    \begin{tabular}{rl}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{stock-icon.png}}
      \end{picture}
      \begin{tabular}{l}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
    \end{tabular}
  }
  \fancyhead[OR]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{stock-icon.png}}
      \end{picture}
    \end{tabular}
  }
\else
  \fancyhead[R]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{stock-icon.png}}
      \end{picture}
    \end{tabular}
  }
\fi

\if@twoside
  % Even pages (left side)
  \fancyhead[EL]{
    \bfseries
    \begin{tabular}{rl}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{stock-icon.png}}
      \end{picture}
      \begin{tabular}{l}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
    \end{tabular}
  }

  % Odd pages (right side)
  \fancyhead[OR]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{stock-icon.png}}
      \end{picture}
    \end{tabular}
  }
\else
  \fancyhead[R]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{stock-icon.png}}
      \end{picture}
    \end{tabular}
  }
\fi