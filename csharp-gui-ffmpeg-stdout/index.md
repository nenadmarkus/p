---
title: How to read FFmpeg output in CSharp
date: 2021-10-31
excerpt: A non-blocking way to read FFmpeg output in CSharp.
tags: CSharp, GUI, Avalonia, WPF, FFmpeg, stdout
layout: default
katex: false
---

There seems to be significant confusion in the C\# community regarding spawning a FFmpeg process within a GUI app (e.g., developed with the [Avalonia framework](https://github.com/AvaloniaUI/Avalonia)) and capturing its standard output.
The following articles are top-ranked on Google but do not provide a straightforward solution:

* <https://stackoverflow.com/questions/50680393/c-sharp-process-standardinput-and-standardoutput-with-ffmpeg-freezes>
* <https://stackoverflow.com/questions/46491619/using-ffmpeg-in-c-cant-get-the-ffmpeg-output>
* <https://www.codeproject.com/Questions/492381/StartInfo-RedirectStandardOutp>

The trick is to read from `stderr` (**!!!!**) after setting the redirection flags and do this in a separate process to not block the GUI:

```
Process process = new Process();
process.StartInfo.FileName = "ffmpeg";             // or "ffmpeg.exe" if on MS Windows
process.StartInfo.Arguments = args;                // ffmpeg args go here
process.StartInfo.UseShellExecute = false;
process.StartInfo.RedirectStandardOutput = true;
process.StartInfo.RedirectStandardError = true;
process.Start();

Task.Run(() => {
	// read all text at once and write it to a file
	string output = process.StandardError.ReadToEnd();
	File.WriteAllText("ffmpeg-output.txt", output);
});
```

If needed, it is simple to modify the above snippet so that the output is read in a line-by-line fashion.
