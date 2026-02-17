import { useState, useEffect } from 'react'
import {
  Paper,
  Text,
  Group,
  Badge,
  Stack,
  Box,
  SimpleGrid,
  Loader,
  Center,
  Title,
  Progress,
  Tooltip,
  ActionIcon,
  ScrollArea,
  Code,
  Divider,
  Table,
  ThemeIcon,
  CopyButton,
  rem
} from '@mantine/core'
import {
  IconUser,
  IconUsers,
  IconGenderMale,
  IconGenderFemale,
  IconMoodSmile,
  IconAlertTriangle,
  IconCopy,
  IconCheck,
  IconEye,
  IconCalendar,
  IconActivity
} from '@tabler/icons-react'
import type { ScenePersonsData, PersonSummary, PersonTimelineEntry } from '../types'

const EMOTION_COLORS: Record<string, string> = {
  happy: 'green',
  sad: 'indigo',
  angry: 'red',
  fear: 'yellow',
  surprise: 'grape',
  disgust: 'teal',
  neutral: 'gray',
  unknown: 'dark'
}

const GENDER_ICONS: Record<string, React.ReactNode> = {
  Male: <IconGenderMale size={14} />,
  Female: <IconGenderFemale size={14} />,
  Man: <IconGenderMale size={14} />,
  Woman: <IconGenderFemale size={14} />
}

interface PersonsPanelProps {
  sceneName: string
  onClose?: () => void
}

export function PersonsPanel({ sceneName }: PersonsPanelProps) {
  const [data, setData] = useState<ScenePersonsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPersonsData()
  }, [sceneName])

  const fetchPersonsData = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/scene/${sceneName}/persons`)
      if (!response.ok) {
        throw new Error(`Failed to fetch persons data: ${response.statusText}`)
      }
      const result = await response.json()
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  // Generate VLM context string for export
  const generateVLMContext = (): string => {
    if (!data) return ''
    
    let context = `Demographics (Per-Person):\n`
    context += `  Total persons tracked: ${data.uniquePersonSlots}\n`
    context += `  Max persons in frame: ${data.maxPersonsInFrame}\n\n`
    
    data.personSummaries.forEach((person, idx) => {
      const ageStr = person.mean_age 
        ? `~${Math.round(person.mean_age)} years old` 
        : 'age unknown'
      const genderStr = person.dominant_gender || 'gender unknown'
      const frameStr = `${person.appearances}/${data.totalFramesAnalyzed} frames`
      
      context += `  Person ${idx + 1}: ${ageStr}, ${genderStr}, appears in ${frameStr}\n`
      
      if (person.dominant_emotion) {
        const emotionParts = Object.entries(person.emotion_distribution)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([emotion, count]) => {
            const pct = Math.round((count / person.appearances) * 100)
            return `${emotion} (${pct}%)`
          })
        context += `    - Emotions: ${emotionParts.join(', ')}\n`
      }
      
      if (person.dominant_race) {
        context += `    - Ethnicity: ${person.dominant_race}\n`
      }
      
      if (person.needs_review_ratio > 0.3) {
        context += `    - Note: Model disagreement in ${Math.round(person.needs_review_ratio * 100)}% of frames\n`
      }
      
      context += '\n'
    })
    
    return context
  }

  if (loading) {
    return (
      <Center h={200}>
        <Stack align="center" gap="md">
          <Loader size="md" color="cerulean" />
          <Text c="dimmed" size="sm">Loading persons data...</Text>
        </Stack>
      </Center>
    )
  }

  if (error) {
    return (
      <Center h={200}>
        <Stack align="center" gap="md">
          <IconAlertTriangle size={32} color="#ff6b6b" />
          <Text c="red" size="sm">{error}</Text>
        </Stack>
      </Center>
    )
  }

  if (!data || data.uniquePersonSlots === 0) {
    return (
      <Center h={200}>
        <Stack align="center" gap="md">
          <IconUsers size={32} color="#6c757d" />
          <Text c="dimmed" size="sm">No persons detected in this scene</Text>
        </Stack>
      </Center>
    )
  }

  return (
    <Stack gap="md">
      {/* Header Stats */}
      <Paper p="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        <Group justify="space-between" align="center">
          <Group gap="lg">
            <Group gap="xs">
              <ThemeIcon size="lg" variant="light" color="cerulean">
                <IconUsers size={20} />
              </ThemeIcon>
              <Box>
                <Text size="xl" fw={600}>{data.uniquePersonSlots}</Text>
                <Text size="xs" c="dimmed">Unique Persons</Text>
              </Box>
            </Group>
            
            <Divider orientation="vertical" />
            
            <Group gap="xs">
              <ThemeIcon size="lg" variant="light" color="grape">
                <IconEye size={20} />
              </ThemeIcon>
              <Box>
                <Text size="xl" fw={600}>{data.totalFramesAnalyzed}</Text>
                <Text size="xs" c="dimmed">Frames Analyzed</Text>
              </Box>
            </Group>
            
            <Divider orientation="vertical" />
            
            <Group gap="xs">
              <ThemeIcon size="lg" variant="light" color="teal">
                <IconActivity size={20} />
              </ThemeIcon>
              <Box>
                <Text size="xl" fw={600}>{data.maxPersonsInFrame}</Text>
                <Text size="xs" c="dimmed">Max in Frame</Text>
              </Box>
            </Group>
          </Group>
          
          <CopyButton value={generateVLMContext()}>
            {({ copied, copy }) => (
              <Tooltip label={copied ? 'Copied!' : 'Copy VLM Context'}>
                <ActionIcon 
                  variant={copied ? 'filled' : 'light'} 
                  color={copied ? 'green' : 'cerulean'}
                  size="lg"
                  onClick={copy}
                >
                  {copied ? <IconCheck size={18} /> : <IconCopy size={18} />}
                </ActionIcon>
              </Tooltip>
            )}
          </CopyButton>
        </Group>
      </Paper>

      {/* Person Summary Cards */}
      <SimpleGrid cols={{ base: 1, sm: 2, md: data.uniquePersonSlots > 2 ? 3 : 2 }} spacing="md">
        {data.personSummaries.map((person, idx) => (
          <PersonCard key={person.person_idx} person={person} index={idx} totalFrames={data.totalFramesAnalyzed} />
        ))}
      </SimpleGrid>

      {/* Timeline Table */}
      <Paper p="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        <Title order={5} mb="md">
          <Group gap="xs">
            <IconCalendar size={18} />
            Frame-by-Frame Timeline
          </Group>
        </Title>
        <ScrollArea h={300}>
          <TimelineTable timeline={data.timeline} uniquePersons={data.uniquePersonSlots} />
        </ScrollArea>
      </Paper>

      {/* VLM Context Preview */}
      <Paper p="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        <Group justify="space-between" mb="sm">
          <Title order={5}>VLM Context Output</Title>
          <CopyButton value={generateVLMContext()}>
            {({ copied, copy }) => (
              <ActionIcon 
                variant="subtle" 
                color={copied ? 'green' : 'gray'}
                size="sm"
                onClick={copy}
              >
                {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
              </ActionIcon>
            )}
          </CopyButton>
        </Group>
        <Code block style={{ whiteSpace: 'pre-wrap', fontSize: rem(11) }}>
          {generateVLMContext()}
        </Code>
      </Paper>
    </Stack>
  )
}

interface PersonCardProps {
  person: PersonSummary
  index: number
  totalFrames: number
}

function PersonCard({ person, index, totalFrames }: PersonCardProps) {
  const coveragePercent = Math.round((person.appearances / totalFrames) * 100)
  const needsReview = person.needs_review_ratio > 0.3
  
  return (
    <Paper p="md" bg="dark.7" withBorder style={{ borderColor: '#2c2e33' }}>
      <Stack gap="sm">
        {/* Header */}
        <Group justify="space-between" align="flex-start">
          <Group gap="sm">
            <ThemeIcon 
              size="xl" 
              radius="xl" 
              variant="gradient" 
              gradient={{ from: 'cerulean', to: 'grape', deg: 135 }}
            >
              <IconUser size={24} />
            </ThemeIcon>
            <Box>
              <Text fw={600}>Person {index + 1}</Text>
              <Text size="xs" c="dimmed">Slot {person.person_idx}</Text>
            </Box>
          </Group>
          
          {needsReview && (
            <Tooltip label={`Model disagreement in ${Math.round(person.needs_review_ratio * 100)}% of frames`}>
              <Badge color="yellow" variant="light" leftSection={<IconAlertTriangle size={12} />}>
                Review
              </Badge>
            </Tooltip>
          )}
        </Group>

        {/* Demographics */}
        <Group gap="xs" wrap="wrap">
          {person.mean_age && (
            <Tooltip label={person.age_range ? `Range: ${person.age_range[0]}-${person.age_range[1]}` : 'Single detection'}>
              <Badge variant="light" color="blue">
                ~{Math.round(person.mean_age)} years
              </Badge>
            </Tooltip>
          )}
          
          {person.dominant_gender && (
            <Badge 
              variant="light" 
              color={person.dominant_gender.toLowerCase().includes('male') ? 'cyan' : 'pink'}
              leftSection={GENDER_ICONS[person.dominant_gender]}
            >
              {person.dominant_gender}
            </Badge>
          )}
          
          {person.dominant_race && (
            <Badge variant="light" color="gray">
              {person.dominant_race}
            </Badge>
          )}
        </Group>

        {/* Emotion */}
        {person.dominant_emotion && (
          <Group gap="xs">
            <IconMoodSmile size={14} color="#6c757d" />
            <Badge 
              variant="dot" 
              color={EMOTION_COLORS[person.dominant_emotion] || 'gray'}
            >
              {person.dominant_emotion}
            </Badge>
            {Object.entries(person.emotion_distribution)
              .filter(([e]) => e !== person.dominant_emotion)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 2)
              .map(([emotion, count]) => (
                <Text key={emotion} size="xs" c="dimmed">
                  {emotion}: {count}
                </Text>
              ))
            }
          </Group>
        )}

        {/* Appearance Coverage */}
        <Box>
          <Group justify="space-between" mb={4}>
            <Text size="xs" c="dimmed">Frame Coverage</Text>
            <Text size="xs" c="dimmed">{person.appearances}/{totalFrames} ({coveragePercent}%)</Text>
          </Group>
          <Progress 
            value={coveragePercent} 
            color="cerulean" 
            size="sm" 
            radius="xl"
          />
        </Box>

        {/* Gender Distribution (if varied) */}
        {Object.keys(person.gender_distribution).length > 1 && (
          <Box>
            <Text size="xs" c="dimmed" mb={4}>Gender Detection Variance</Text>
            <Group gap={4}>
              {Object.entries(person.gender_distribution).map(([gender, count]) => (
                <Badge key={gender} size="xs" variant="outline" color="gray">
                  {gender}: {count}
                </Badge>
              ))}
            </Group>
          </Box>
        )}
      </Stack>
    </Paper>
  )
}

interface TimelineTableProps {
  timeline: PersonTimelineEntry[]
  uniquePersons: number
}

function TimelineTable({ timeline, uniquePersons }: TimelineTableProps) {
  // Group by frame
  const frameGroups = timeline.reduce((acc, entry) => {
    if (!acc[entry.frame_idx]) {
      acc[entry.frame_idx] = []
    }
    acc[entry.frame_idx].push(entry)
    return acc
  }, {} as Record<number, PersonTimelineEntry[]>)
  
  const frames = Object.keys(frameGroups).map(Number).sort((a, b) => a - b)
  
  return (
    <Table striped highlightOnHover withTableBorder>
      <Table.Thead>
        <Table.Tr>
          <Table.Th style={{ width: 80 }}>Frame</Table.Th>
          <Table.Th style={{ width: 60 }}>#</Table.Th>
          {Array.from({ length: uniquePersons }, (_, i) => (
            <Table.Th key={i}>Person {i + 1}</Table.Th>
          ))}
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {frames.map(frameIdx => {
          const entries = frameGroups[frameIdx]
          const personMap = entries.reduce((acc, e) => {
            acc[e.person_idx] = e
            return acc
          }, {} as Record<number, PersonTimelineEntry>)
          
          return (
            <Table.Tr key={frameIdx}>
              <Table.Td>
                <Code>{frameIdx}</Code>
              </Table.Td>
              <Table.Td>
                <Badge size="sm" variant="light" color="gray">
                  {entries.length}
                </Badge>
              </Table.Td>
              {Array.from({ length: uniquePersons }, (_, personIdx) => {
                const entry = personMap[personIdx]
                if (!entry) {
                  return (
                    <Table.Td key={personIdx}>
                      <Text size="xs" c="dimmed">—</Text>
                    </Table.Td>
                  )
                }
                
                return (
                  <Table.Td key={personIdx}>
                    <Group gap={4} wrap="nowrap">
                      {entry.age && (
                        <Badge size="xs" variant="light" color="blue">
                          {Math.round(entry.age)}y
                        </Badge>
                      )}
                      {entry.gender && (
                        <Badge 
                          size="xs" 
                          variant="light" 
                          color={entry.gender.toLowerCase().includes('male') ? 'cyan' : 'pink'}
                        >
                          {entry.gender.charAt(0)}
                        </Badge>
                      )}
                      {entry.emotion && (
                        <Badge 
                          size="xs" 
                          variant="dot" 
                          color={EMOTION_COLORS[entry.emotion] || 'gray'}
                        >
                          {entry.emotion.slice(0, 3)}
                        </Badge>
                      )}
                      {entry.needs_review && (
                        <Tooltip label="Model disagreement">
                          <IconAlertTriangle size={12} color="#fab005" />
                        </Tooltip>
                      )}
                    </Group>
                  </Table.Td>
                )
              })}
            </Table.Tr>
          )
        })}
      </Table.Tbody>
    </Table>
  )
}
