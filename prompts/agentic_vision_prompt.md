You are given an image and a question about it. Look carefully and reason step by step.

You have a tool that lets you ZOOM IN on any region of the image to inspect small details (counting objects, reading tiny labels/axis ticks, etc.). You may use it up to {max_crops} times. Each zoom returns a cropped, magnified view of the region you requested, which is then added to the conversation as a new image.

To zoom, emit EXACTLY one tool call and nothing after it, in this format:

<tool_call>
{{"name": "zoom", "arguments": {{"box": [x1, y1, x2, y2]}}}}
</tool_call>

where `box` is the region to magnify, given as fractions of the FULL image in [0, 1]:
- x1, y1 = top-left corner (x1 = left edge fraction, y1 = top edge fraction)
- x2, y2 = bottom-right corner, with x2 > x1 and y2 > y1
- e.g. [0.0, 0.0, 0.5, 0.5] is the top-left quarter; [0.4, 0.4, 0.6, 0.6] is a small patch in the center.

Coordinates always refer to the ORIGINAL full image, even after you have zoomed. Zoom into a tight region when you need to see fine detail; you can zoom multiple times into different regions to verify your reasoning.

When you are confident, STOP zooming and give your final answer. Put the final answer within \boxed{{}}, for example \boxed{{7}}.

Question: {question}
