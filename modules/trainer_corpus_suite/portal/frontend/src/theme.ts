import { createTheme, MantineColorsTuple, rem } from '@mantine/core'

// Cerulean Blue scale
const cerulean: MantineColorsTuple = [
  '#e6f4ff',
  '#cce5fc',
  '#99cbf9',
  '#66b0f6',
  '#3395f3',
  '#007bff',
  '#0069d9',
  '#0057b3',
  '#00458c',
  '#003366'
]

// Dark Grey scale
const slate: MantineColorsTuple = [
  '#f8f9fa',
  '#e9ecef',
  '#dee2e6',
  '#ced4da',
  '#adb5bd',
  '#6c757d',
  '#495057',
  '#343a40',
  '#212529',
  '#0d1117'
]

export const theme = createTheme({
  colors: {
    cerulean,
    slate,
    dark: [
      '#C1C2C5',
      '#A6A7AB',
      '#909296',
      '#5c5f66',
      '#373A40',
      '#2C2E33',
      '#25262b',
      '#1A1B1E',
      '#141517',
      '#0d1117'
    ]
  },
  primaryColor: 'cerulean',
  primaryShade: 5,
  fontFamily: '"Outfit", -apple-system, BlinkMacSystemFont, sans-serif',
  fontFamilyMonospace: '"JetBrains Mono", monospace',
  headings: {
    fontFamily: '"Outfit", sans-serif',
    fontWeight: '600'
  },
  defaultRadius: 'md',
  cursorType: 'pointer',
  components: {
    Button: {
      defaultProps: {
        variant: 'filled'
      }
    },
    Paper: {
      defaultProps: {
        shadow: 'sm'
      }
    },
    Table: {
      styles: {
        th: {
          fontWeight: 600,
          fontSize: rem(13),
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }
      }
    }
  }
})
