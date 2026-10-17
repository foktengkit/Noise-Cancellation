param([string]$Output = "test-artifacts/clean-speech.wav")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$path = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($path)) | Out-Null
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $speaker.SetOutputToWaveFile($path, $format)
    $voice = $speaker.Voice.Name
    $speaker.Speak("Please check this voice recording. The fan is running in the background. We want to keep the words clear and reduce the noise. This is a test of speech enhancement, not active noise cancellation.")
    Write-Output "Voice: $voice"
    Write-Output "Generated: $path"
} finally {
    $speaker.Dispose()
}
