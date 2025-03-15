// import { VoiceSentimentAnalysisComponent } from "@/components/frontend-src-components-voice-sentiment-analysis"

// export default function Page() {
//   return <VoiceSentimentAnalysisComponent />
// }
import { VoiceSentimentAnalysisComponent } from "@/components/frontend-src-components-voice-sentiment-analysis"
import { ThemeProvider } from "@/components/theme-provider"

export default function Home() {
  return (
    <ThemeProvider>
      <main className="min-h-screen p-4">
        <h1 className="text-3xl font-bold text-center mb-8">Customer Engagement Sentiment Tracking System</h1>
        <VoiceSentimentAnalysisComponent />
      </main>
    </ThemeProvider>
  )
}

