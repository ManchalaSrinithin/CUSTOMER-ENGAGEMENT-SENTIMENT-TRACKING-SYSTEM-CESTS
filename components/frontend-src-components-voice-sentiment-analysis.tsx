"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Loader2, Upload, Moon, Sun, Star, Mic, Square, Clock, ChevronRight } from "lucide-react"
import { useTheme } from "next-themes"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"

interface RecordingHistory {
  id: string
  filename: string
  duration: number
  timestamp: string
  transcription: string
  sentiment: string
  confidence: number
  Compound: number
  Negative:number
  Neutral : number
  Positive : number
  summary:string

}

interface SentimentAnalysis {
  sentiment: string
  score: number
  polarity: number
  vader_scores: {
    compound: number
    neg: number
    neu: number
    pos: number
  }
  summary: string
}

export function VoiceSentimentAnalysisComponent() {
  const [text, setText] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysis, setAnalysis] = useState<SentimentAnalysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [audioURL, setAudioURL] = useState<string | null>(null)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [isSaving, setIsSaving] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [history, setHistory] = useState<RecordingHistory[]>([])
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<string | null>(null)
  const [apiAvailable, setApiAvailable] = useState(true)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    checkApiAvailability()
    fetchHistory()
    return () => {
      // Cleanup
      if (audioURL) {
        URL.revokeObjectURL(audioURL)
      }
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }, [])

  const checkApiAvailability = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/health", {
        method: "GET",
        headers: { Accept: "application/json" },
        // Short timeout to avoid long waits
        signal: AbortSignal.timeout(2000),
      })
      setApiAvailable(response.ok)
      if (!response.ok) {
        showStatus("API server is not available. Using local storage for history.", true)
      }
    } catch (error) {
      console.error("API health check failed:", error)
      setApiAvailable(false)
      showStatus("API server is not available. Using local storage for history.", true)
    }
  }

  const [isOnline, setIsOnline] = useState(true)

  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    setIsOnline(navigator.onLine)
    window.addEventListener("online", handleOnline)
    window.addEventListener("offline", handleOffline)

    return () => {
      window.removeEventListener("online", handleOnline)
      window.removeEventListener("offline", handleOffline)
    }
  }, [])

  const fetchHistory = async () => {
    setIsLoadingHistory(true)
    try {
      // Try to fetch from the API
      if (apiAvailable) {
        try {
          const response = await fetch("http://127.0.0.1:5000/recordings")
          if (response.ok) {
            const data = await response.json()
            setHistory(data)
            return
          }
        } catch (apiError) {
          console.error("API error:", apiError)
          // Continue to fallback if API fails
        }
      }

      // Fallback: Check if we have any saved recordings in localStorage
      const savedRecordings = localStorage.getItem("voiceSentimentRecordings")
      if (savedRecordings) {
        setHistory(JSON.parse(savedRecordings))
      } else {
        // If no saved recordings, set empty array
        setHistory([])
      }

      showStatus("Using locally stored recordings history", false)
    } catch (err) {
      console.error("Error fetching history:", err)
      showStatus("Failed to load recording history", true)
      // Set empty array on error
      setHistory([])
    } finally {
      setIsLoadingHistory(false)
    }
  }

  const loadHistoryItem = async (id: string) => {
    try {
      setSelectedHistoryItem(id)

      // Check if it's a local recording (starts with "local_")
      if (id.startsWith("local_")) {
        // Find the recording in our history state
        const recording = history.find((item) => item.id === id)
        if (recording) {
          setText(recording.transcription || "")
          setAnalysis({
            sentiment: recording.sentiment,
            score: recording.confidence,
            polarity: 0, // Default value
            vader_scores: {
              compound: recording.Compound,
              neg: recording.Negative,
              neu: recording.Neutral,
              pos: recording.Positive,
            },
            summary: recording.summary,
          })
          showStatus(`Loaded local recording from ${new Date(recording.timestamp).toLocaleString()}`)
          return
        }
      }

      // If not local or not found locally, try the API
      if (apiAvailable) {
        try {
          const response = await fetch(`http://127.0.0.1:5000/recordings/${id}`)
          if (response.ok) {
            const data = await response.json()

            // Update the form with the recording data
            setText(data.transcription || "")
            setAnalysis({
              sentiment: data.sentiment,
              score: data.confidence,
              polarity: data.polarity,
              vader_scores: {
                compound: data.compound,
                neg: data.negative,
                neu: data.neutral,
                pos: data.positive,
              },
              summary: data.summary,
            })

            // If there's an audio file, set it up
            if (data.file_path) {
              try {
                const audioResponse = await fetch(`http://127.0.0.1:5000/audio-file/${id}`)
                if (audioResponse.ok) {
                  const blob = await audioResponse.blob()
                  const url = URL.createObjectURL(blob)
                  setAudioURL(url)
                }
              } catch (audioError) {
                console.error("Error loading audio file:", audioError)
              }
            }

            showStatus(`Loaded recording from ${new Date(data.timestamp).toLocaleString()}`)
            return
          }
        } catch (apiError) {
          console.error("API error when loading recording:", apiError)
        }
      }

      // If we get here, we couldn't load the recording
      showStatus("Could not load recording details", true)
    } catch (err) {
      console.error("Error loading recording:", err)
      showStatus("Failed to load recording", true)
    }
  }

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      // Clear any previous recording
      if (audioURL) {
        URL.revokeObjectURL(audioURL)
        setAudioURL(null)
      }
    }
  }

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  const showStatus = (message: string, isError = false) => {
    setStatusMessage(message)
    setTimeout(() => setStatusMessage(null), 5000)
    console.log(isError ? `Error: ${message}` : message)
  }

  const startRecording = async () => {
    try {
      // Reset previous recording data
      if (audioURL) {
        URL.revokeObjectURL(audioURL)
        setAudioURL(null)
      }
      setFile(null)
      setRecordingDuration(0)
      setAnalysis(null)
      setText("")
      setSelectedHistoryItem(null)

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const options = { mimeType: "audio/webm" }
      const mediaRecorder = new MediaRecorder(stream, options)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" })
        const url = URL.createObjectURL(audioBlob)
        setAudioURL(url)

        const audioFile = new File([audioBlob], `recording_${Date.now()}.webm`, {
          type: "audio/webm",
          lastModified: Date.now(),
        })
        setFile(audioFile)

        // Automatically process the recording
        await processRecording(audioFile)
      }

      // Start recording
      mediaRecorder.start()
      setIsRecording(true)

      // Start timer for recording duration
      timerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1)
      }, 1000)

      showStatus("Recording started. Speak clearly into your microphone.")
    } catch (err) {
      console.error("Error accessing microphone:", err)
      setError("Could not access microphone. Please check permissions.")
      showStatus("Could not access microphone. Please check permissions.", true)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)

      // Stop all tracks
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop())

      // Clear timer
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }

      showStatus(`Processing audio (${formatTime(recordingDuration)})...`)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  // Process the recording - save to database and analyze
  const processRecording = async (audioFile: File) => {
    if (!audioFile) {
      showStatus("No audio recording found to process", true)
      return
    }

    setIsSaving(true)
    setIsAnalyzing(true)
    setError(null)

    try {
      if (!isOnline) {
        throw new Error("No internet connection. Please check your network and try again.")
      }

      // First, send directly to the audio analysis endpoint
      const formData = new FormData()
      formData.append("audio", audioFile)

      // Send to your audio analysis endpoint
      const response = await fetch("http://127.0.0.1:5000/audio", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorText = await response.text()
        let errorMessage = `Server error: ${response.status}`
        try {
          const errorData = JSON.parse(errorText)
          if (errorData.message) {
            errorMessage = errorData.message
          }
        } catch (e) {
          // If parsing fails, use the status code message
        }
        throw new Error(errorMessage)
      }

      // Process the response
      const data = await response.json()
      setAnalysis(data.analysis_result)
      setText(data.transcription)

      // Now save to database in the background
      const dbFormData = new FormData()
      dbFormData.append("audio", audioFile)
      dbFormData.append("timestamp", new Date().toISOString())
      dbFormData.append("duration", recordingDuration.toString())
      dbFormData.append("transcription", data.transcription || recordingDuration.toString())
      dbFormData.append("transcription", data.transcription || "")

      // Optional: Add sentiment data to database record
      if (data.analysis_result) {
        dbFormData.append("sentiment", data.analysis_result.sentiment || "")
        dbFormData.append("confidence", String(data.analysis_result.score || 0))
      }

      // Send to your database endpoint
      if (apiAvailable) {
        try {
          const dbResponse = await fetch("http://127.0.0.1:5000/save-recording", {
            method: "POST",
            body: dbFormData,
          })

          if (!dbResponse.ok) {
            console.error("Warning: Failed to save to database:", await dbResponse.text())
            showStatus("Analysis complete, but database save failed", true)
          } else {
            const dbResult = await dbResponse.json()
            showStatus(`Recording processed and saved. ID: ${dbResult.recordingId || "Unknown"}`)

            // Save to localStorage as a backup
            try {
              // Save to localStorage as a backup
              const newRecording = {
                id: dbResult.recordingId || `local_${Date.now()}`,
                filename: audioFile.name,
                duration: recordingDuration,
                timestamp: new Date().toISOString(),
                transcription: data.transcription || "",
                sentiment: data.analysis_result?.sentiment || "Unknown",
                confidence: data.analysis_result?.score || 0,
                Compound: data.analysis_result?.vader_scores.compound.toFixed(2) || 0,
                Negative: data.analysis_result?.vader_scores.neg.toFixed(2) || 0,
                Neutral: data.analysis_result?.vader_scores.neu.toFixed(2) || 0,
                Positive: data.analysis_result?.vader_scores.pos.toFixed(2) || 0,
                summary:data.analysis_result?.summary,
                

              }

              // Get existing recordings
              const existingRecordings = localStorage.getItem("voiceSentimentRecordings")
              const recordings = existingRecordings ? JSON.parse(existingRecordings) : []

              // Add new recording to the beginning
              recordings.unshift(newRecording)

              // Save back to localStorage
              localStorage.setItem("voiceSentimentRecordings", JSON.stringify(recordings))

              // Update state
              setHistory([newRecording, ...history])
            } catch (localStorageError) {
              console.error("Failed to save to localStorage:", localStorageError)
            }

            // Refresh the history list
            fetchHistory()
          }
        } catch (dbError) {
          console.error("Database error:", dbError)
          saveToLocalStorageOnly(audioFile, data)
        }
      } else {
        // If API is not available, save to localStorage only
        saveToLocalStorageOnly(audioFile, data)
      }
    } catch (err) {
      console.error("Processing error:", err)
      setError(err instanceof Error ? err.message : "An unexpected error occurred. Please try again.")
      showStatus(err instanceof Error ? err.message : "Failed to process recording", true)
    } finally {
      setIsSaving(false)
      setIsAnalyzing(false)
    }
  }

  // Helper function to save to localStorage only
  const saveToLocalStorageOnly = (audioFile: File, data: any) => {
    console.log(data)
    try {
      const newRecording = {
        id: `local_${Date.now()}`,
        filename: audioFile.name,
        duration: recordingDuration,
        timestamp: new Date().toISOString(),
        transcription: data.transcription || "",
        sentiment: data.analysis_result?.sentiment || "Unknown",
        confidence: data.analysis_result?.score || 0,
        Compound: data.analysis_result?.vader_scores.compound.toFixed(2) || 0,
        Negative: data.analysis_result?.vader_scores.neg.toFixed(2) || 0,
        Neutral: data.analysis_result?.vader_scores.neu.toFixed(2) || 0,
        Positive: data.analysis_result?.vader_scores.pos.toFixed(2) || 0,
        summary:data.analysis_result?.summary,
      }

      // Get existing recordings
      const existingRecordings = localStorage.getItem("voiceSentimentRecordings")
      const recordings = existingRecordings ? JSON.parse(existingRecordings) : []

      // Add new recording to the beginning
      recordings.unshift(newRecording)

      // Save back to localStorage
      localStorage.setItem("voiceSentimentRecordings", JSON.stringify(recordings))

      // Update state
      setHistory([newRecording, ...history])
      showStatus("Recording saved locally (API unavailable)")
    } catch (localStorageError) {
      console.error("Failed to save to localStorage:", localStorageError)
      showStatus("Failed to save recording", true)
    }
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement> | null) => {
    if (e) e.preventDefault()
    setIsAnalyzing(true)
    setError(null)
    setSelectedHistoryItem(null)

    try {
      if (!isOnline) {
        throw new Error("No internet connection. Please check your network and try again.")
      }

      const formData = new FormData()

      if (file) {
        formData.append("audio", file)
        const response = await fetch("http://127.0.0.1:5000/audio", {
          method: "POST",
          body: formData,
        })

        if (!response.ok) {
          const errorText = await response.text()
          let errorMessage = `Server error: ${response.status}`
          try {
            const errorData = JSON.parse(errorText)
            if (errorData.message) {
              errorMessage = errorData.message
            }
          } catch (e) {
            // If parsing fails, use the status code message
          }
          throw new Error(errorMessage)
        }

        // Make sure to extract the data from the response
        const data = await response.json()
        setAnalysis(data.analysis_result)
        setText(data.transcription)

        // Save to database
        if (apiAvailable) {
          const dbFormData = new FormData()
          dbFormData.append("audio", file)
          dbFormData.append("timestamp", new Date().toISOString())
          dbFormData.append("transcription", data.transcription || "")

          if (data.analysis_result) {
            dbFormData.append("sentiment", data.analysis_result.sentiment || "")
            dbFormData.append("confidence", String(data.analysis_result.score || 0))
          }

          try {
            const dbResponse = await fetch("http://127.0.0.1:5000/save-recording", {
              method: "POST",
              body: dbFormData,
            })

            if (dbResponse.ok) {
              fetchHistory()
            }
          } catch (dbError) {
            console.error("Database error:", dbError)
            saveToLocalStorageOnly(file, data)
          }
        } else {
          saveToLocalStorageOnly(file, data)
        }
      } else if (text) {
        formData.append("text", text)
        const response = await fetch("http://127.0.0.1:5000/transcribe", {
          method: "POST",
          body: formData,
        })

        if (!response.ok) {
          const errorText = await response.text()
          let errorMessage = `Server error: ${response.status}`
          try {
            const errorData = JSON.parse(errorText)
            if (errorData.message) {
              errorMessage = errorData.message
            }
          } catch (e) {
            // If parsing fails, use the status code message
          }
          throw new Error(errorMessage)
        }

        // Make sure to extract the data from the response
        const data = await response.json()
        setAnalysis(data.analysis_result)
        setText(data.transcription)
      } else {
        throw new Error("Please provide either text, file, or record audio to analyze.")
      }
    } catch (err) {
      console.error("Analysis error:", err)
      setError(err instanceof Error ? err.message : "An unexpected error occurred. Please try again.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment?.toLowerCase()) {
      case "positive":
        return "text-green-500"
      case "negative":
        return "text-red-500"
      case "neutral":
      default:
        return "text-yellow-500"
    }
  }

  const getStarRating = (score: number) => {
    const percentage = score * 100
    if (percentage >= 80) return 5
    if (percentage >= 60) return 4
    if (percentage >= 40) return 3
    if (percentage >= 20) return 2
    return 1
  }

  const renderStars = (rating: number) => {
    return Array(5)
      .fill(0)
      .map((_, i) => (
        <Star key={i} className={`h-5 w-5 ${i < rating ? "text-yellow-400 fill-yellow-400" : "text-gray-300"}`} />
      ))
  }

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString()
    } catch (e) {
      return dateString
    }
  }

  if (!mounted) return null

  return (
    <div className="container mx-auto p-4">
      <div className="flex justify-end mb-4">
        <Button variant="outline" size="icon" onClick={toggleTheme} className="rounded-full">
          {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Side - Input Section */}
        <div className="space-y-4">
          <Card className="w-full">
            <CardHeader>
              <CardTitle className="text-xl font-bold">Audio Transcription & Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              {statusMessage && (
                <div
                  className={`mb-4 p-3 rounded-md ${statusMessage.includes("Error") || statusMessage.includes("failed") ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-200" : "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-200"}`}
                >
                  {statusMessage}
                </div>
              )}

              <form onSubmit={(e) => handleSubmit(e)} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="text-input">Audio Transcription:</Label>
                  <Textarea
                    id="text-input"
                    placeholder="The analyzed audio transcription is shown here..."
                    value={text}
                    onChange={handleTextChange}
                    className="min-h-[150px]"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Record Audio</Label>
                  <div className="flex flex-col space-y-2">
                    <div className="flex items-center gap-2">
                      {!isRecording ? (
                        <Button
                          type="button"
                          onClick={startRecording}
                          variant="outline"
                          className="w-full"
                          disabled={isAnalyzing || isSaving}
                        >
                          <Mic className="mr-2 h-4 w-4" />
                          Start Recording
                        </Button>
                      ) : (
                        <Button type="button" onClick={stopRecording} variant="destructive" className="w-full">
                          <Square className="mr-2 h-4 w-4" />
                          Stop Recording ({formatTime(recordingDuration)})
                        </Button>
                      )}
                    </div>

                    {audioURL && (
                      <div className="mt-2">
                        <audio src={audioURL} controls className="w-full" />
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="file-input">Or Upload an Audio File</Label>
                  <Input
                    id="file-input"
                    type="file"
                    onChange={handleFileChange}
                    accept="audio/*"
                    disabled={isRecording || isAnalyzing || isSaving}
                  />
                </div>

                <Button
                  type="submit"
                  className="w-full"
                  disabled={isAnalyzing || !isOnline || isSaving || isRecording || (!text && !file)}
                >
                  {isAnalyzing || isSaving ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {isAnalyzing ? "Analyzing..." : "Saving..."}
                    </>
                  ) : !isOnline ? (
                    "Offline - Check Connection"
                  ) : (
                    <>
                      <Upload className="mr-2 h-4 w-4" />
                      Analyze Sentiment
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
            <CardFooter>
              {error && (
                <Alert variant="destructive" className="w-full">
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardFooter>
          </Card>
        </div>

        {/* Right Side - Results and History */}
        <div className="space-y-4">
          {/* Top 40% - Analysis Results */}
          <Card className="w-full">
            <CardHeader>
              <CardTitle className="text-xl font-bold">Analysis Results</CardTitle>
            </CardHeader>
            <CardContent>
              {analysis ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">Sentiment:</span>
                    <span className={`font-bold ${getSentimentColor(analysis.sentiment)}`}>{analysis.sentiment}</span>
                  </div>

                  <div>
                    <span className="font-semibold">Confidence:</span>
                    <div className="flex items-center mt-1">
                      {renderStars(getStarRating(analysis.score))}
                      <span className="ml-2 text-sm">({(analysis.score * 100).toFixed(2)}%)</span>
                    </div>
                  </div>

                  <div>
                    <span className="font-semibold">Summary:</span>
                    <p className="mt-1 text-sm">{analysis.summary}</p>
                  </div>

                  <div>
                    <span className="font-semibold">Vader Scores:</span>
                    <div className="grid grid-cols-2 gap-2 mt-1">
                      {analysis?.vader_scores && (
                        <>
                          <div className="text-sm">Compound: {analysis.vader_scores.compound}</div>
                          <div className="text-sm">Negative: {analysis.vader_scores.neg}</div>
                          <div className="text-sm">Neutral: {analysis.vader_scores.neu}</div>
                          <div className="text-sm">Positive: {analysis.vader_scores.pos}</div>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No analysis results to display. Record or upload audio to analyze.
                </div>
              )}
            </CardContent>
          </Card>

          {/* Bottom 60% - History */}
          <Card className="w-full">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="text-xl font-bold">Recording History</CardTitle>
                <Button variant="outline" size="sm" onClick={fetchHistory} disabled={isLoadingHistory}>
                  {isLoadingHistory ? <Loader2 className="h-4 w-4 animate-spin" /> : <Clock className="h-4 w-4" />}
                  <span className="ml-2">Refresh</span>
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px] pr-4">
                {isLoadingHistory ? (
                  <div className="flex justify-center items-center h-full">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : history.length > 0 ? (
                  <div className="space-y-2">
                    {history.map((item) => (
                      <div key={item.id}>
                        <div
                          className={`flex items-center justify-between p-3 rounded-md cursor-pointer hover:bg-muted transition-colors ${selectedHistoryItem === item.id ? "bg-muted" : ""}`}
                          onClick={() => loadHistoryItem(item.id)}
                        >
                          <div className="flex-1">
                            <div className="font-medium flex items-center">
                              <span className={`w-2 h-2 rounded-full mr-2 ${getSentimentColor(item.sentiment)}`}></span>
                              {item.transcription ? (
                                item.transcription.length > 40 ? (
                                  item.transcription.substring(0, 40) + "..."
                                ) : (
                                  item.transcription
                                )
                              ) : (
                                <span className="text-muted-foreground italic">No transcription</span>
                              )}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                              {formatDate(item.timestamp)} • {item.sentiment}
                            </div>
                          </div>
                          <ChevronRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <Separator className="my-2" />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No recording history found. Record or upload audio to create history.
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

