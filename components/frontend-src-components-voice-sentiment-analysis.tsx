"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Loader2, Upload, Moon, Sun, Mic, Square } from 'lucide-react'
import { useTheme } from 'next-themes'

export function VoiceSentimentAnalysisComponent() {
  const [text, setText] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysis, setAnalysis] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const [isOnline, setIsOnline] = useState(true)

  useEffect(() => {
    setIsOnline(navigator.onLine)
    window.addEventListener('online', () => setIsOnline(true))
    window.addEventListener('offline', () => setIsOnline(false))

    return () => {
      window.removeEventListener('online', () => setIsOnline(true))
      window.removeEventListener('offline', () => setIsOnline(false))
    }
  }, [])

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark')
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const options = { mimeType: 'audio/webm' }
      const mediaRecorder = new MediaRecorder(stream, options)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const audioFile = new File([audioBlob], "recorded_audio.webm", { type: 'audio/webm' })
        setFile(audioFile)
        await handleSubmit(null)
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      console.error('Error accessing microphone:', err)
      setError('Could not access microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop())
    }
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement> | null) => {
    if (e) e.preventDefault()
    setIsAnalyzing(true)
    setError(null)

    try {
      if (!isOnline) {
        throw new Error("No internet connection. Please check your network and try again.")
      }

      const formData = new FormData()
      
      if (file) {
        formData.append('audio', file)
        const response = await fetch('http://127.0.0.1:5000/audio', {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.message || `Server error: ${response.status}`)
        }

        const data = await response.json()
        setAnalysis(data.analysis_result)
        setText(data.transcription)
      } else if (text) {
        formData.append('text', text)
        const response = await fetch('http://127.0.0.1:5000/transcribe', {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.message || `Server error: ${response.status}`)
        }

        const data = await response.json()
        setAnalysis(data.analysis_result)
        setText(data.transcription)
      } else {
        throw new Error("Please provide either text, file, or record audio to analyze.")
      }
    } catch (err) {
      console.error('Analysis error:', err)
      setError(err instanceof Error ? err.message : 'An unexpected error occurred. Please try again.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment?.toLowerCase()) {
      case 'positive':
        return 'text-green-500'
      case 'negative':
        return 'text-red-500'
      case 'neutral':
      default:
        return 'text-yellow-500'
    }
  }

  if (!mounted) return null

  return (
    <div className="container mx-auto p-4 max-w-2xl">
      <div className="flex justify-end mb-4">
        <Button
          variant="outline"
          size="icon"
          onClick={toggleTheme}
          className="rounded-full"
        >
          {theme === 'dark' ? (
            <Sun className="h-5 w-5" />
          ) : (
            <Moon className="h-5 w-5" />
          )}
        </Button>
      </div>

      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center">Customer Engagement Sentiment Tracking Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="text-input">The Audio transcription: </Label>
              <Textarea
                id="text-input"
                placeholder="The Analyzed Audio transcription is shown Here.. "
                value={text}
                onChange={handleTextChange}
                className="min-h-[100px]"
              />
            </div>

            

            <div className="space-y-2">
              <Label htmlFor="file-input">Upload an audio file</Label>
              <Input
                id="file-input"
                type="file"
                onChange={handleFileChange}
                accept="audio/*"
              />
            </div>

            <Button type="submit" className="w-full" disabled={isAnalyzing || !isOnline}>
              {isAnalyzing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : !isOnline ? (
                'Offline - Check Connection'
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Analyze Sentiment
                </>
              )}
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex flex-col space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {analysis && (
            <div className="w-full space-y-2">
              <h3 className="font-semibold">Analysis Results:</h3>
              {/* <p><strong>Analyzed Text:</strong> {analysis.text}</p> */}
              <p>
                <strong>Sentiment:</strong>
                <span className={`ml-2 font-bold ${getSentimentColor(analysis.sentiment)}`}>
                  {analysis.sentiment}
                </span>
              </p>
              <div>
                <strong>Confidence:</strong>
                <Progress value={analysis.score * 100} className="mt-2" />
                <p className="text-right text-sm">{(analysis.score * 100).toFixed(2)}%</p>
              </div>
              
              <div>
                <strong>Summary:</strong> {analysis.summary}
              </div>
              <div>
                <strong>Vader Scores:</strong>
                <ul>
                  {analysis?.vader_scores ? (
                    <>
                      <li>Compound: {analysis.vader_scores.compound}</li>
                      <li>Negative: {analysis.vader_scores.neg}</li>
                      <li>Neutral: {analysis.vader_scores.neu}</li>
                      <li>Positive: {analysis.vader_scores.pos}</li>
                    </>
                  ) : (
                    <li>No Vader Scores available</li>
                  )}
                </ul>
              </div>
            </div>
          )}
        </CardFooter>
      </Card>
    </div>
  )
}