import { useState, useEffect } from 'react'
import {
  AppShell,
  Container,
  Tabs,
  Title,
  Text,
  Group,
  Badge,
  Loader,
  Center,
  Stack,
  Box,
  rem
} from '@mantine/core'
import {
  IconVideo,
  IconScissors,
  IconChartBar,
  IconDatabase,
  IconCircleCheck,
  IconAlertCircle,
  IconChartDots3,
  IconPhoto,
  IconVolume,
} from '@tabler/icons-react'
import { ClipsTable } from './components/ClipsTable'
import { VideosTable } from './components/VideosTable'
import { InsightsDashboard } from './components/InsightsDashboard'
import { DataExplorer } from './components/DataExplorer'
import { CaptionReviewer } from './components/CaptionReviewer'
import { AudioExplorer } from './components/AudioExplorer'
import type { ClipData, VideoData, PipelineInsights, DemographicsData } from './types'

function App() {
  const [clips, setClips] = useState<ClipData[]>([])
  const [videos, setVideos] = useState<VideoData[]>([])
  const [insights, setInsights] = useState<PipelineInsights | null>(null)
  const [demographics, setDemographics] = useState<DemographicsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string | null>('insights')
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      // Fetch pipeline data and demographics in parallel
      const [pipelineRes, demoRes] = await Promise.all([
        fetch('/api/pipeline-data'),
        fetch('/api/demographics').catch(() => null)
      ])
      
      if (!pipelineRes.ok) throw new Error('Failed to fetch pipeline data')
      const data = await pipelineRes.json()
      setClips(data.clips)
      setVideos(data.videos)
      setInsights(data.insights)
      
      // Demographics is optional
      if (demoRes?.ok) {
        const demoData = await demoRes.json()
        setDemographics(demoData)
      }
      
      setLastUpdated(new Date())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AppShell padding="md">
      <Container size="xl" py="lg">
        {/* Header */}
        <Box mb="xl">
          <Group justify="space-between" align="flex-start">
            <Stack gap={4}>
              <Group gap="sm">
                <IconDatabase size={32} stroke={1.5} color="#3395f3" />
                <Title
                  order={1}
                  style={{
                    background: 'linear-gradient(135deg, #fff 0%, #3395f3 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    fontWeight: 700
                  }}
                >
                  Pipeline Data Portal
                </Title>
              </Group>
              <Text c="dimmed" size="sm" ml={rem(44)}>
                Video Analysis Pipeline • Training Data Explorer
              </Text>
            </Stack>
            
            <Group gap="xs">
              {loading ? (
                <Badge leftSection={<Loader size={12} color="cerulean" />} variant="light" color="cerulean">
                  Loading...
                </Badge>
              ) : error ? (
                <Badge leftSection={<IconAlertCircle size={14} />} variant="light" color="red">
                  Error
                </Badge>
              ) : (
                <Badge
                  leftSection={<IconCircleCheck size={14} />}
                  variant="light"
                  color="green"
                  className="pulse-badge"
                >
                  Connected
                </Badge>
              )}
              {lastUpdated && (
                <Text size="xs" c="dimmed">
                  Updated: {lastUpdated.toLocaleTimeString()}
                </Text>
              )}
            </Group>
          </Group>
        </Box>

        {/* Content */}
        {loading ? (
          <Center h={400}>
            <Stack align="center" gap="md">
              <Loader size="lg" color="cerulean" type="dots" />
              <Text c="dimmed">Loading pipeline data...</Text>
            </Stack>
          </Center>
        ) : error ? (
          <Center h={400}>
            <Stack align="center" gap="md">
              <IconAlertCircle size={48} color="#ff6b6b" />
              <Text c="red" fw={500}>{error}</Text>
              <Text c="dimmed" size="sm">
                Make sure the backend server is running on port 8088
              </Text>
            </Stack>
          </Center>
        ) : (
          <Tabs value={activeTab} onChange={setActiveTab} variant="outline">
            <Tabs.List mb="lg">
              <Tabs.Tab
                value="insights"
                leftSection={<IconChartBar size={16} />}
              >
                Insights Dashboard
              </Tabs.Tab>
              <Tabs.Tab
                value="explorer"
                leftSection={<IconChartDots3 size={16} />}
                rightSection={
                  demographics ? (
                    <Badge size="sm" variant="filled" color="grape">
                      {demographics.summary.totalPersonsDetected}
                    </Badge>
                  ) : null
                }
              >
                Data Explorer
              </Tabs.Tab>
              <Tabs.Tab
                value="clips"
                leftSection={<IconScissors size={16} />}
                rightSection={
                  <Badge size="sm" variant="filled" color="cerulean">
                    {clips.length}
                  </Badge>
                }
              >
                Clips
              </Tabs.Tab>
              <Tabs.Tab
                value="videos"
                leftSection={<IconVideo size={16} />}
                rightSection={
                  <Badge size="sm" variant="filled" color="cerulean">
                    {videos.length}
                  </Badge>
                }
              >
                Videos
              </Tabs.Tab>
              <Tabs.Tab
                value="captions"
                leftSection={<IconPhoto size={16} />}
              >
                Caption Reviewer
              </Tabs.Tab>
              <Tabs.Tab
                value="audio"
                leftSection={<IconVolume size={16} />}
              >
                Audio Analysis
              </Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="insights">
              {insights && <InsightsDashboard insights={insights} clips={clips} />}
            </Tabs.Panel>

            <Tabs.Panel value="explorer">
              <DataExplorer demographics={demographics} clips={clips} />
            </Tabs.Panel>

            <Tabs.Panel value="clips">
              <ClipsTable clips={clips} />
            </Tabs.Panel>

            <Tabs.Panel value="videos">
              <VideosTable videos={videos} />
            </Tabs.Panel>

            <Tabs.Panel value="captions">
              <CaptionReviewer />
            </Tabs.Panel>

            <Tabs.Panel value="audio">
              <AudioExplorer clips={clips} />
            </Tabs.Panel>
          </Tabs>
        )}
      </Container>
    </AppShell>
  )
}

export default App
