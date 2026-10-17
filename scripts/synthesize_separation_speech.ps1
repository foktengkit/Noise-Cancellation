param([string]$OutputDirectory = "test-artifacts/separation")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$directory = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
[System.IO.Directory]::CreateDirectory($directory) | Out-Null
$clips = @(
    @{Name="speaker-a.wav"; Voice="Microsoft Hanhan Desktop"; Text="Please keep my voice clear. This sentence belongs to the first speaker. We are testing two overlapping voices."},
    @{Name="speaker-b.wav"; Voice="Microsoft Zira Desktop"; Text="Another person is speaking at the same time. Today the weather is pleasant and the train arrives at noon."}
)
foreach ($clip in $clips) {
    $speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        $speaker.SelectVoice($clip.Voice)
        $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
        $speaker.SetOutputToWaveFile((Join-Path $directory $clip.Name), $format)
        $speaker.Speak($clip.Text)
        Write-Output ("Generated " + $clip.Name + " with " + $clip.Voice)
    } finally {
        $speaker.Dispose()
    }
}
