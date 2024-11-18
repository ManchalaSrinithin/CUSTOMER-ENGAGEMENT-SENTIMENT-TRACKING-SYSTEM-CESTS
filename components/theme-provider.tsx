"use client"

import * as React from "react"
import { ThemeProvider as NextThemesProvider } from "next-themes"

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextThemesProvider
      defaultTheme="system" // or "light" / "dark" based on your requirement
      enableSystem={true} // To allow automatic system-based theme
    >
      {children}
    </NextThemesProvider>
  )
}
